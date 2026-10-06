"""Serialize a grid2op ObsData into a text prompt for ms-swift's training pipeline.

ms-swift's GRPOTrainer expects text prompts. The graph encoder + vllm embedding
injection path is reserved for Phase 2; for Phase 1 we represent the grid state
as a compact JSON string inserted into the prompt.
"""
from __future__ import annotations

import json

import numpy as np

from gridzero.env.observation import ObsData

_PROMPT_TEMPLATE = """\
<|im_start|>system
You are an autonomous power grid operator. Given the current grid state, \
output a single JSON tool call to control the grid. Output only valid JSON, nothing else.
<|im_end|>
<|im_start|>user
{grid_state}
<|im_end|>
<|im_start|>assistant
"""


def obs_to_prompt(obs: ObsData) -> str:
    """Convert an ObsData into a chat-formatted prompt string.

    Serializes key electrical quantities as a compact JSON object.
    The full flat vector is not included — only the most decision-relevant fields.
    """
    state = _extract_key_fields(obs)
    grid_state_str = json.dumps(state, separators=(",", ":"))
    return _PROMPT_TEMPLATE.format(grid_state=grid_state_str)


def obs_to_dataset_row(obs: ObsData, env_id: int = 0) -> dict:
    """Convert an ObsData to a dict row suitable for a HuggingFace Dataset.

    The 'prompt' field is consumed by ms-swift's GRPOTrainer.
    Additional fields are passed through to the ORM reward function as kwargs.
    """
    return {
        "prompt": obs_to_prompt(obs),
        # Serialized flat vector lets the ORM reconstruct the observation
        # for environment stepping without needing a live env reference.
        "obs_flat": obs.flat.tolist(),
        "n_lines": obs.n_lines,
        "n_loads": obs.n_loads,
        "n_gens": obs.n_gens,
        "n_substations": obs.n_substations,
        "env_id": env_id,
    }


def _extract_key_fields(obs: ObsData) -> dict:
    """Extract the most decision-relevant fields from the flat observation vector.

    Full feature engineering lives here — add substation topology, generator
    ramp limits, etc. as the model matures.
    """
    # These attribute names match grid2op's Observation API.
    # We gracefully skip any that the specific environment doesn't expose.
    source = obs.raw if obs.raw is not None else obs
    fields: dict = {}

    _try_add(fields, "rho", source, "rho")            # line loading ratio
    _try_add(fields, "line_status", source, "line_status")
    _try_add(fields, "load_p", source, "load_p")
    _try_add(fields, "gen_p", source, "gen_p")
    _try_add(fields, "v_or", source, "v_or")

    fields["n_lines"] = obs.n_lines
    fields["n_loads"] = obs.n_loads
    fields["n_gens"] = obs.n_gens
    return fields


def _try_add(fields: dict, key: str, source: object, attr: str) -> None:
    """Add obs.attr to fields as a rounded list, silently skip if absent."""
    arr = getattr(source, attr, None)
    if arr is not None:
        if isinstance(arr, np.ndarray):
            fields[key] = [round(float(v), 4) for v in arr]
        else:
            fields[key] = arr
