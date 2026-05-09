"""GRPOTrainer subclass that injects observation embeddings instead of text."""
from __future__ import annotations

import json
import re
import torch
from datasets import Dataset
from omegaconf import DictConfig
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import selective_log_softmax, entropy_from_logits
from trl.trainer.utils import pad

from gridzero.encoder.flat_encoder import FlatObsEncoder
from gridzero.env.observation import ObsData
from gridzero.training.embed_prompt import (
    _get_embed_fn,
    build_generation_embeds,
    build_training_embeds,
    cache_template_ids,
)
from gridzero.training.env import GridEnv
from gridzero.training.gspo import SYSTEM_PROMPT
from gridzero.training.reward import grid_reward

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_and_execute(text: str, env: GridEnv) -> None:
    """Parse a generated tool call and execute it on the environment."""
    m = _TOOL_CALL_RE.search(text)
    if m is None:
        env.do_nothing()
        return
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        env.do_nothing()
        return
    name = call.get("name", "do_nothing")
    args = call.get("arguments", {})
    method = getattr(env, name, None)
    if method is None:
        env.do_nothing()
        return
    try:
        method(**args)
    except (RuntimeError, ValueError, TypeError, IndexError):
        pass


class EmbeddingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that encodes observations as embeddings for vLLM generation.

    Instead of serializing observations as JSON text, a learned FlatObsEncoder
    maps the observation vector into the model's embedding space. These
    embeddings replace the user-message content in the prompt and are passed
    to vLLM via ``EmbedsPrompt``.
    """

    def __init__(
        self,
        model: str,
        args: GRPOConfig,
        train_dataset: Dataset,
        encoder_cfg: DictConfig,
        env_name: str = "l2rpn_case14_sandbox",
        reward_funcs=None,
        **kwargs,
    ):
        reward_funcs = reward_funcs or grid_reward
        # Do NOT pass environment_factory — we manage environments ourselves
        kwargs.pop("environment_factory", None)

        # Patch VLLMGeneration._init_vllm to inject enable_prompt_embeds=True
        # into the LLM constructor, so CUDA graphs and buffers are set up
        # correctly from the start (avoids costly re-creation).
        from trl.generation.vllm_generation import VLLMGeneration
        _original_init_vllm = VLLMGeneration._init_vllm

        def _patched_init_vllm(vg_self):
            from vllm import LLM as _LLM
            _original_LLM_init = _LLM.__init__

            def _llm_init_with_embeds(llm_self, *a, **kw):
                kw["enable_prompt_embeds"] = True
                return _original_LLM_init(llm_self, *a, **kw)

            _LLM.__init__ = _llm_init_with_embeds
            try:
                _original_init_vllm(vg_self)
            finally:
                _LLM.__init__ = _original_LLM_init

        VLLMGeneration._init_vllm = _patched_init_vllm
        try:
            super().__init__(
                model=model,
                args=args,
                train_dataset=train_dataset,
                reward_funcs=reward_funcs,
                **kwargs,
            )
        finally:
            VLLMGeneration._init_vllm = _original_init_vllm

        device = self.accelerator.device
        hidden_size = self.model.config.hidden_size

        # Initialize observation encoder — reset once to determine flat_dim
        probe_env = GridEnv(env_name=env_name)
        probe_env.reset()
        flat_dim = probe_env.last_obs_data.flat.shape[0]
        del probe_env
        self.obs_encoder = FlatObsEncoder(encoder_cfg, flat_dim=flat_dim)
        assert self.obs_encoder.output_dim == hidden_size, (
            f"Encoder output_dim ({self.obs_encoder.output_dim}) must match "
            f"model hidden_size ({hidden_size})"
        )
        self.obs_encoder = self.obs_encoder.to(device)
        if self.args.bf16:
            self.obs_encoder = self.obs_encoder.to(torch.bfloat16)

        # Create environments (same pattern as TRL's __init__)
        gen_batch_size = self.args.per_device_train_batch_size * self.args.steps_per_generation
        self.environments = [GridEnv(env_name=env_name) for _ in range(gen_batch_size)]

        # Cache chat template token IDs
        self._prefix_ids, self._suffix_ids = cache_template_ids(
            self.processing_class, SYSTEM_PROMPT
        )


    def create_optimizer(self):
        optimizer = super().create_optimizer()
        optimizer.add_param_group({
            "params": list(self.obs_encoder.parameters()),
            "lr": self.args.learning_rate,
        })
        return optimizer

    def _generate_and_score_completions(
        self, inputs: list[dict]
    ) -> dict[str, torch.Tensor]:
        from vllm import SamplingParams

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # Step 1: Reset environments and encode observations
        obs_data_list: list[ObsData] = []
        for inp, env in zip(inputs, self.environments):
            chronics_id = inp.get("chronics_id", 0)
            env.reset(chronics_id=chronics_id)
            obs_data_list.append(env.last_obs_data)

        # Step 2: Encode observations (no grad during generation)
        embed_fn = _get_embed_fn(self.model)
        prompt_embeds_list: list[torch.Tensor] = []
        with torch.no_grad():
            for obs_data in obs_data_list:
                obs_emb = self.obs_encoder(obs_data)
                prompt_emb = build_generation_embeds(
                    self._prefix_ids, self._suffix_ids, obs_emb, embed_fn,
                )
                prompt_embeds_list.append(prompt_emb)

        # Step 3: Generate with vLLM using EmbedsPrompt
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        generation_kwargs = {
            "n": 1,
            "temperature": self.vllm_generation.temperature,
            "top_p": self.vllm_generation.top_p,
            "top_k": self.vllm_generation.top_k,
            "min_p": 0.0 if self.vllm_generation.min_p is None else self.vllm_generation.min_p,
            "max_tokens": self.vllm_generation.max_completion_length,
            "logprobs": self.vllm_generation.logprobs,
            "repetition_penalty": self.vllm_generation.repetition_penalty,
        }
        generation_kwargs.update(self.vllm_generation.generation_kwargs)

        import vllm as _vllm
        from packaging.version import Version
        if Version(_vllm.__version__) <= Version("0.10.2"):
            from vllm.sampling_params import GuidedDecodingParams as StructuredOutputsParams
            so_key = "guided_decoding"
        else:
            from vllm.sampling_params import StructuredOutputsParams
            so_key = "structured_outputs"

        if isinstance(generation_kwargs.get(so_key), dict):
            generation_kwargs[so_key] = StructuredOutputsParams(**generation_kwargs[so_key])

        sampling_params = SamplingParams(**generation_kwargs)

        vllm_prompts = [{"prompt_embeds": emb} for emb in prompt_embeds_list]
        all_outputs = self.vllm_generation.llm.generate(
            vllm_prompts, sampling_params=sampling_params, use_tqdm=False,
        )

        completion_ids_list = [list(out.outputs[0].token_ids) for out in all_outputs]

        # Extract top-1 logprobs from vLLM output
        sampling_logps_list = None
        if all_outputs[0].outputs[0].logprobs is not None:
            sampling_logps_list = []
            for out in all_outputs:
                seq_logps = []
                for lp_dict in out.outputs[0].logprobs:
                    top_item = min(lp_dict.values(), key=lambda x: x.rank)
                    seq_logps.append(top_item.logprob)
                sampling_logps_list.append(seq_logps)

        # Decode completions for reward function
        completions_text = self.processing_class.batch_decode(
            [list(ids) for ids in completion_ids_list], skip_special_tokens=False,
        )

        # Step 4: Execute tool calls and get rewards
        for text, env in zip(completions_text, self.environments):
            _parse_and_execute(text, env)

        rewards = grid_reward(environments=self.environments)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Step 5: Compute advantages (GRPO group normalization)
        grouped = rewards_tensor.view(-1, num_generations)
        mean_rewards = grouped.mean(dim=1).repeat_interleave(num_generations)
        if num_generations > 1:
            std_rewards = grouped.std(dim=1).repeat_interleave(num_generations)
        else:
            std_rewards = torch.zeros_like(rewards_tensor)
        advantages = (rewards_tensor - mean_rewards) / (std_rewards + 1e-4)

        # Step 6: Build output tensors
        prompt_len = len(self._prefix_ids) + self.obs_encoder._seq_len + len(self._suffix_ids)
        dummy_prompt_ids = torch.zeros(len(inputs), prompt_len, dtype=torch.long, device=device)
        prompt_mask = torch.ones_like(dummy_prompt_ids)

        completion_ids = [torch.tensor(list(ids), device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if sampling_logps_list and sampling_logps_list[0]:
            sampling_per_token_logps = [
                torch.tensor(logps, device=device) for logps in sampling_logps_list
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        num_items_in_batch = completion_mask.sum()

        # Log metrics
        self._metrics[mode]["reward"].append(rewards_tensor.mean().item())
        self._metrics[mode]["reward_std"].append(rewards_tensor.std().item())
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(rewards_tensor.mean().item())
            self._metrics[mode][f"rewards/{name}/std"].append(rewards_tensor.std().item())

        prompts_text = ["[embedding prompt]"] * len(inputs)
        self._logs["prompt"].extend(prompts_text)
        self._logs["completion"].extend(completions_text)
        for name in self.reward_func_names:
            self._logs["rewards"][name].extend(rewards)
        self._logs["advantages"].extend(advantages.tolist())

        output = {
            "prompt_ids": dummy_prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "cached_obs_data": obs_data_list,
        }
        if sampling_per_token_logps is not None:
            output["old_per_token_logps"] = sampling_per_token_logps
        return output

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        inputs_embeds=None,
        completion_token_ids=None,
        **kwargs,
    ):
        """Compute log-probs, with optional inputs_embeds path for embedding training."""
        if inputs_embeds is None:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, compute_entropy=compute_entropy, **kwargs,
            )

        batch_size = batch_size or inputs_embeds.size(0)
        all_logps = []
        all_entropies = []

        for start in range(0, inputs_embeds.size(0), batch_size):
            embeds_batch = inputs_embeds[start:start + batch_size]
            mask_batch = attention_mask[start:start + batch_size]
            comp_ids_batch = completion_token_ids[start:start + batch_size]

            model_inputs = {
                "inputs_embeds": embeds_batch,
                "attention_mask": mask_batch,
                "use_cache": False,
            }
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            logps = selective_log_softmax(logits, comp_ids_batch.to(logits.device))
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _compute_loss(self, model, inputs):
        if "cached_obs_data" not in inputs:
            return super()._compute_loss(model, inputs)

        model_device = next(model.parameters()).device
        completion_ids = inputs["completion_ids"].to(model_device)
        completion_mask = inputs["completion_mask"].to(model_device)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask

        embed_fn = _get_embed_fn(model)

        # Re-encode observations WITH gradients for encoder training
        inputs_embeds_list = []
        for i, obs_data in enumerate(inputs["cached_obs_data"]):
            obs_emb = self.obs_encoder(obs_data)
            embs, _ = build_training_embeds(
                self._prefix_ids, self._suffix_ids,
                obs_emb, completion_ids[i], embed_fn,
            )
            inputs_embeds_list.append(embs)

        # Pad to same length
        max_len = max(e.size(0) for e in inputs_embeds_list)
        hidden_size = inputs_embeds_list[0].size(1)
        inputs_embeds = torch.zeros(
            len(inputs_embeds_list), max_len, hidden_size,
            device=inputs_embeds_list[0].device, dtype=inputs_embeds_list[0].dtype,
        )
        attention_mask = torch.zeros(
            len(inputs_embeds_list), max_len,
            device=inputs_embeds_list[0].device, dtype=torch.long,
        )
        for i, embs in enumerate(inputs_embeds_list):
            inputs_embeds[i, :embs.size(0)] = embs
            attention_mask[i, :embs.size(0)] = 1

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids=None, attention_mask=attention_mask,
            logits_to_keep=logits_to_keep, compute_entropy=True,
            inputs_embeds=inputs_embeds, completion_token_ids=completion_ids,
        )

        # GRPO clipped policy gradient loss
        advantages = inputs["advantages"].to(model_device)
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is not None:
            old_per_token_logps = old_per_token_logps.to(model_device)
        else:
            old_per_token_logps = per_token_logps.detach()

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        per_token_loss = per_token_loss * mask
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()

        current_gradient_accumulation_steps = self.args.gradient_accumulation_steps
        if hasattr(self, "accelerator") and hasattr(self.accelerator, "gradient_accumulation_steps"):
            current_gradient_accumulation_steps = self.accelerator.gradient_accumulation_steps
        loss = loss / current_gradient_accumulation_steps

        # Log metrics
        mode = "train" if self.model.training else "eval"
        mean_kl = None
        if self.beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            mean_kl = ((per_token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean().item()

        clip_ratio = (
            (coef_1 < (1 - self.epsilon_low)) | (coef_1 > (1 + self.epsilon_high))
        ).float()
        clip_ratio = ((clip_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean().item()

        self._metrics[mode]["loss"].append(loss.item() * current_gradient_accumulation_steps)
        self._metrics[mode]["clip_ratio"].append(clip_ratio)
        if mean_kl is not None:
            self._metrics[mode]["kl"].append(mean_kl)
        if entropies is not None:
            mean_entropy = ((entropies * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean().item()
            self._metrics[mode]["entropy"].append(mean_entropy)

        return loss
