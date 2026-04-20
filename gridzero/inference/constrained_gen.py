"""Constrained generation utilities."""
from __future__ import annotations

from gridzero.env.actions import get_json_schema


def build_guided_json_params(action_space=None) -> dict:
    """Return the guided_json extra_body params for a vllm chat request.

    Args:
        action_space: Optional grid2op action_space. If provided, element ID
                      ranges in the schema are tightened to only valid indices
                      (future: prevents the model from generating actions for
                      non-existent lines or generators).

    Returns:
        Dict with a 'guided_json' key for vllm's extra_body.
    """
    schema = get_json_schema()

    if action_space is not None:
        # TODO: narrow integer ranges — e.g. line_id maximum = n_line - 1
        pass

    return {"guided_json": schema}
