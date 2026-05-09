"""Inference parameter wiring tests."""

from gridzero.inference.constrained_gen import build_guided_json_params


def test_guided_json_returns_schema_root():
    schema = build_guided_json_params()
    assert isinstance(schema, dict)
    assert "guided_json" not in schema
    # discriminated union marker should be present
    assert "action_type" in str(schema)
