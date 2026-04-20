"""Sequence log-probability extraction from a HuggingFace causal LM."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def sequence_log_prob(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    encoder_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute log P(completion | prompt, encoder_embeddings).

    Concatenates prompt + completion tokens, runs one forward pass, then
    sums the per-token log-probs over the completion span only.

    Args:
        model: The causal LM (policy or reference policy).
        prompt_ids: Prompt token IDs, shape [B, T_prompt].
        completion_ids: Completion token IDs, shape [B, T_comp].
        attention_mask: Optional mask for the full sequence [B, T_prompt + T_comp].
        encoder_embeddings: Optional [N, d_model] prefix embeddings.
                            If provided, these are prepended to the input embeddings
                            before the prompt tokens (requires a patched model or
                            custom forward — TODO for vllm training integration).

    Returns:
        Scalar per-sequence log-probs, shape [B].
    """
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # [B, T_prompt + T_comp]

    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)

    logits = outputs.logits  # [B, T, vocab_size]

    # Shift: predict token t+1 from logits at position t
    shift_logits = logits[:, :-1, :]          # [B, T-1, V]
    shift_ids = full_ids[:, 1:]               # [B, T-1]

    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    token_log_probs = log_probs_all.gather(
        dim=-1, index=shift_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, T-1]

    # Sum only over the completion portion
    T_prompt = prompt_ids.shape[1]
    completion_log_probs = token_log_probs[:, T_prompt - 1:]  # [B, T_comp]
    return completion_log_probs.sum(dim=-1)  # [B]
