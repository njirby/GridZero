#!/usr/bin/env python3
"""Generate grid2op Harbor task directories from the template.

Each task = one (chronic, horizon, seed) episode. Params are delivered to the
container via literal [environment.env] values in task.toml (Harbor passes
literals through unchanged), so ALL tasks share the prebuilt gridzero-rl-base
image — no per-task image build.

Usage:
  # one task (smoke)
  python lego/make_tasks.py --out /data/gridzero_tasks --chronic 3 --horizon 24 --seed 7

  # a training pool: chronics 0..31, horizon 24, seed 0
  python lego/make_tasks.py --out /data/gridzero_tasks --chronic 0-31 --horizon 24 --seed 0

Then build the index with Lego-RL's own tool:
  lego-rl/.venv/bin/python lego-rl/utils/create_task_index.py \
      --tasks_dir /data/gridzero_tasks --output /data/gridzero_index --split train
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "task_template"


def _expand(spec: str) -> list[int]:
    """'3' -> [3]; '0-31' -> [0..31]; '0,5,9' -> [0,5,9]; '0-3,7' -> [0,1,2,3,7]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def render(template: Path, **kw: str) -> str:
    # string.Template ($-placeholders) so TOML's own {braces} are left alone.
    return Template(template.read_text(encoding="utf-8")).substitute(**kw)


def make_one(out_root: Path, chronic: int, horizon: int, seed: int, agent_timeout: float) -> Path:
    instance_id = f"c{chronic}_h{horizon}_s{seed}"
    task_dir = out_root / f"gridzero_{instance_id}"
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    kw = dict(chronic=chronic, horizon=horizon, seed=seed,
              instance_id=instance_id, agent_timeout=int(agent_timeout))
    (task_dir / "instruction.md").write_text(render(TEMPLATE / "instruction.md", **kw), encoding="utf-8")
    (task_dir / "task.toml").write_text(render(TEMPLATE / "task.toml", **kw), encoding="utf-8")
    shutil.copyfile(TEMPLATE / "tests" / "test.sh", task_dir / "tests" / "test.sh")
    (task_dir / "tests" / "test.sh").chmod(0o755)
    return task_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root for task dirs")
    ap.add_argument("--chronic", default="0", help="chronic id(s): '3' | '0-31' | '0,5,9'")
    ap.add_argument("--horizon", type=int, default=24, help="episode horizon (steps)")
    ap.add_argument("--seed", default="0", help="seed(s): '0' | '0,1'")
    ap.add_argument("--agent-timeout", type=float, default=1800.0, help="agent phase timeout (s)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    chronic_ids = _expand(args.chronic)
    seed_ids = _expand(args.seed)

    made = []
    for c in chronic_ids:
        for s in seed_ids:
            made.append(make_one(out_root, c, args.horizon, s, args.agent_timeout))

    print(f"generated {len(made)} task(s) under {out_root}")
    for t in made[:10]:
        print(f"  {t.name}")
    if len(made) > 10:
        print(f"  ... and {len(made) - 10} more")


if __name__ == "__main__":
    main()
