import json, os
import jsonschema
import pytest
from backend.app.sim_session import SimSession

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = json.load(open(os.path.join(ROOT, "contracts", "C3-grid-state.schema.json")))


@pytest.fixture(scope="module")
def sess():
    s = SimSession()
    s.reset()
    return s


def test_reset_c3_valid(sess):
    c3 = sess.latest_state()
    jsonschema.validate(c3, SCHEMA)
    assert c3["t"] == 0
    assert c3["cum_reward"] == 0.0          # reset reward excluded
    assert c3["n_line"] == 20 and c3["n_sub"] == 14
    assert c3["last_action"] is None


def test_metadata_valid(sess):
    m = sess.metadata()
    assert m["n_sub"] == 14 and len(m["subs"]) == 14
    assert m["n_line"] == 20 and len(m["lines"]) == 20
    assert all("x" in s and "y" in s for s in m["subs"])


def test_act_flips_line_and_c3_still_valid(sess):
    name = str(sess._env.name_line[1])  # 0_4_1
    out, verb, err = sess.act({"set_line_status": {name: -1}})
    assert err is None, f"act rejected: {err}"
    c3 = sess.latest_state()
    jsonschema.validate(c3, SCHEMA)
    line = [l for l in c3["lines"] if l["name"] == name][0]
    assert line["status"] == "down"
    assert c3["n_down"] >= 1


def test_reward_semantics(sess):
    before = sess.latest_state()["cum_reward"]
    out, verb = sess.step(1)
    after = sess.latest_state()["cum_reward"]
    assert after > before  # one completed step added its (~64) reward
    assert "cum_reward" in out


def test_illegal_action_returns_error(sess):
    out, verb, err = sess.act({"set_line_status": {"99_99_9": -1}})
    assert err is not None
    assert out["illegal"] is True
    assert verb["is_illegal"] is True


def test_render_writes_png(sess, tmp_path):
    d = sess.render(width=800, out="test_render")
    assert os.path.exists(d["path"]) and os.path.getsize(d["path"]) > 0
