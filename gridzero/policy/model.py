"""Randomly-initialized Qwen3-style transformer policy."""
from __future__ import annotations

from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


def build_policy(cfg: DictConfig) -> PreTrainedModel:
    """Instantiate a randomly-initialized Qwen3 model.

    Loads the architecture config from HuggingFace (matching cfg.model_name dims)
    but does NOT download or load any pretrained weights — the model starts from
    random initialization so GSPO trains it tabula rasa.

    Args:
        cfg: Policy config with keys: model_name, (optionally) d_model, n_layers, n_heads.

    Returns:
        Randomly initialized causal LM ready for GSPO training.
    """
    config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)

    # Override arch hyperparameters if explicitly set in config
    if cfg.get("d_model"):
        config.hidden_size = cfg.d_model
    if cfg.get("n_layers"):
        config.num_hidden_layers = cfg.n_layers
    if cfg.get("n_heads"):
        config.num_attention_heads = cfg.n_heads

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model


class GridZeroPolicy:
    """Wrapper around HF model providing save/load and a clean interface for GSPO."""

    def __init__(self, cfg: DictConfig) -> None:
        self.model = build_policy(cfg)
        self.cfg = cfg

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path: str) -> None:
        """Save model weights and config to path."""
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str, cfg: DictConfig) -> "GridZeroPolicy":
        """Load model from a saved checkpoint directory."""
        instance = cls.__new__(cls)
        instance.cfg = cfg
        instance.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True
        )
        return instance
