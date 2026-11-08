"""Rerun button: replay one episode with a given seed, judge it, and print its outcome.

  python rerun.py --seed 3 --condition pressure --n 8
  python rerun.py --seed 3 --analysis-only     # rebuild all tables/figures from saved traces

The environment, the agents' sampling seeds and the judges' seeds are all
derived from --seed. Batched GPU inference is not guaranteed to be bitwise
deterministic, so a fresh rerun can differ from the saved trace in a few
requests; the saved traces in results/ are the record the report is built from,
and `--analysis-only` reproduces every number in the report from them exactly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sandbox.env import EnvConfig
from sandbox.monitors import RULE_MONITORS, overuse_labels
from sandbox.runner import load_trace, run_episode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", default="none",
                    choices=["none", "norm", "pressure", "msg", "fair", "hug", "greedy1", "burst"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default="results/rerun")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    args = ap.parse_args()

    if args.analysis_only:
        return subprocess.call([sys.executable, "analyze.py"])

    out = Path(args.out)
    cfg = EnvConfig(n_agents=args.n, rounds=args.rounds)
    path = run_episode(cfg, args.condition, args.seed, out / "traces")
    if not args.no_judge:
        subprocess.check_call([sys.executable, "judge_traces.py", "--traces", str(out / "traces"),
                               "--out", str(out / "judges"), "--glob", path.name, "--force"])
    run = load_trace(path)
    rounds, env = run["rounds"], run["start"]["env"]
    print(json.dumps(run["end"], indent=1))
    ou = overuse_labels(rounds, env["regen"])
    print("round  budget  used  overuse  " + "  ".join(RULE_MONITORS))
    alarms = {k: f(rounds, env) for k, f in RULE_MONITORS.items()}
    for t, r in enumerate(rounds):
        print(f"{r['round']:5d}  {r['budget_after']:6d}  {r['consumed']:4d}  {ou[t]!s:7s}  "
              + "  ".join(f"{alarms[k][t]!s:6s}" for k in RULE_MONITORS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
