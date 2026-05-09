"""Constrained generation utilities."""
from __future__ import annotations

from gridzero.env.actions import get_json_schema, get_structured_output_regex


def build_guided_json_params(action_space=None) -> dict:
    """Return the guided_json extra_body params for a vllm chat request.

    Args:
        action_space: Optional grid2op action_space. If provided, element ID
                      ranges in the schema are tightened to only valid indices
                      (future: prevents the model from generating actions for
                      non-existent lines or generators).

    Returns:
        JSON schema dictionary for vllm's guided_json.
    """
    schema = get_json_schema()

    if action_space is not None:
        # TODO: narrow integer ranges — e.g. line_id maximum = n_line - 1
        pass

    return schema


def build_outlines_generator(hf_model, tokenizer):
    """Build an outlines regex-constrained generator for HF model evaluation.

    Returns a callable: generator(prompt) -> str (valid ToolCall JSON).
    """
    import outlines

    # outlines_core FSM doesn't support ^ and $ anchors — strip them.
    pattern = get_structured_output_regex().strip("^$")
    model = outlines.from_transformers(hf_model, tokenizer)
    return outlines.Generator(model, outlines.regex(pattern))
