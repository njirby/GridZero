"""Tests for tool call schemas and JSON schema generation."""
import json
import re
import pytest

from gridzero.env.actions import (
    DoNothingAction,
    SetLineStatusAction,
    ChangeBusAction,
    RedispatchAction,
    CurtailAction,
    StorageAction,
    get_json_schema,
    get_structured_output_regex,
)


def test_schema_is_dict():
    schema = get_json_schema()
    assert isinstance(schema, dict)


def test_schema_has_discriminator():
    schema = get_json_schema()
    # Pydantic discriminated union produces oneOf or $defs
    raw = json.dumps(schema)
    assert "action_type" in raw


def test_schema_requires_action_type_in_all_variants():
    schema = get_json_schema()
    defs = schema.get("$defs", {})
    assert isinstance(defs, dict)
    for name, def_schema in defs.items():
        if not isinstance(def_schema, dict):
            continue
        props = def_schema.get("properties", {})
        if "action_type" in props:
            required = def_schema.get("required", [])
            assert "action_type" in required, f"{name} missing required action_type"


def test_do_nothing_roundtrip():
    action = DoNothingAction()
    data = json.loads(action.model_dump_json())
    assert data["action_type"] == "do_nothing"


def test_set_line_status_fields():
    action = SetLineStatusAction(line_id=3, status="disconnect")
    assert action.line_id == 3
    assert action.status == "disconnect"


def test_change_bus_fields():
    action = ChangeBusAction(element_type="load", element_id=0, bus=2)
    assert action.bus == 2


def test_redispatch_fields():
    action = RedispatchAction(gen_id=1, delta_mw=-5.0)
    assert action.delta_mw == pytest.approx(-5.0)


def test_curtail_non_negative():
    with pytest.raises(Exception):
        CurtailAction(gen_id=0, max_mw=-10.0)


def test_storage_charge_discharge():
    charge = StorageAction(storage_id=0, mw=10.0)
    discharge = StorageAction(storage_id=0, mw=-10.0)
    assert charge.mw > 0
    assert discharge.mw < 0


@pytest.mark.parametrize(
    "payload",
    [
        '{"action_type":"do_nothing"}',
        '{"action_type":"set_line_status","line_id":3,"status":"disconnect"}',
        '{"action_type":"change_bus","element_type":"load","element_id":0,"bus":2}',
        '{"action_type":"redispatch","gen_id":1,"delta_mw":-5.0}',
        '{"action_type":"curtail","gen_id":0,"max_mw":10.0}',
        '{"action_type":"storage","storage_id":0,"mw":-1.25}',
    ],
)
def test_structured_output_regex_accepts_valid_payloads(payload: str):
    regex = get_structured_output_regex()
    assert re.fullmatch(regex, payload) is not None


def test_structured_output_regex_rejects_unknown_action():
    regex = get_structured_output_regex()
    assert re.fullmatch(regex, '{"action_type":"totally_fake"}') is None
