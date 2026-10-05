#!/usr/bin/env python
"""De-risking probe for the opencode driver (C5).

Assumes the backend sim is ALREADY running on :8731 (start it with
OPENCODE_DISABLE=1 so it doesn't also spawn opencode). This probe starts
`opencode serve`, creates a session, kicks off a short prompt that calls simctl,
and prints every C4 agent.*/session.status event the driver emits onto the bus.

Run:  ./.venv/bin/python scripts/probe_driver.py
"""
import asyncio, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("OPENCODE_DISABLE", None)

from backend.app.event_bus import EventBus      # noqa: E402
from backend.app.opencode_driver import OpenCodeDriver  # noqa: E402

BUS = EventBus(ep_id="ep-probe")
DRIVER = OpenCodeDriver(BUS, ROOT)
Q = BUS.subscribe()

KICKOFF = (
    "You have a live grid sim. Run `simctl status` and then `simctl observe`. "
    "Report the t value and the maximum line loading. Keep it under 2 sentences."
)
BUDGET = 60.0


async def drain(deadline):
    while time.time() < deadline:
        try:
            f = await asyncio.to_thread(Q.get, True, 1.5)
        except Exception:
            continue
        t = f["type"]
        if t.startswith("agent.") or t == "session.status" or t == "system":
            d = f["data"]
            if t == "agent.delta":
                print(f"  delta[{d.get('field')}] {d.get('delta')!r}")
            elif t == "agent.tool_call":
                print(f"  TOOL_CALL {d.get('tool')}: {str(d.get('input'))[:140]}")
            elif t == "agent.tool_result":
                print(f"  TOOL_RESULT({d.get('status')}) {str(d.get('output'))[:160]}")
            elif t == "agent.turn_end":
                print(f"  TURN_END {d}")
            elif t == "session.status":
                print(f"  STATUS {d.get('status')}")
            else:
                print(f"  {t} {str(d)[:120]}")


async def main():
    await DRIVER.start()
    print("opencode available:", DRIVER.available)
    if not DRIVER.available:
        return
    sid = await DRIVER.create_session()
    print("session:", sid)
    deadline = time.time() + BUDGET
    task = asyncio.create_task(drain(deadline))
    await DRIVER.kickoff(KICKOFF)
    await task
    await DRIVER.stop()
    print("=== probe done ===")


if __name__ == "__main__":
    asyncio.run(main())
