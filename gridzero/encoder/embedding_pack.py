"""Serialize encoder output for vllm multimodal embedding injection."""
from __future__ import annotations

import base64
import io

import torch


def pack_embeddings_for_vllm(embeddings: torch.Tensor) -> dict:
    """Serialize [N, d_model] encoder output to vllm multi_modal_data payload.

    vllm --enable-mm-embeds accepts embeddings encoded as base64 tensors.
    Each embedding in the sequence is a separate entry so vllm can correctly
    map them to placeholder tokens in the prompt.

    Args:
        embeddings: Float tensor of shape [N, d_model].

    Returns:
        Dict suitable for the multi_modal_data field in a vllm chat request.
    """
    assert embeddings.ndim == 2, f"Expected [N, d_model], got {embeddings.shape}"
    buf = io.BytesIO()
    torch.save(embeddings.cpu().contiguous(), buf)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "gridzero_embeddings",
        "data": encoded,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
    }


def unpack_embeddings_from_vllm(payload: dict) -> torch.Tensor:
    """Inverse of pack_embeddings_for_vllm — deserialize base64 payload to tensor."""
    raw = base64.b64decode(payload["data"])
    buf = io.BytesIO(raw)
    return torch.load(buf, weights_only=True)
