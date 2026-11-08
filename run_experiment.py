"""Run a grid of episodes (conditions x N x seeds) and write one JSONL trace per run.

Example:
  python run_experiment.py --conditions none norm pressure msg --n 4 8 --seeds 0 1 2 \
      --rounds 20 --out results/traces
Existing finished traces are skipped, so an interrupted grid can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sandbox.env import EnvConfig
from sandbox.runner import run_episode, run_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["none", "norm", "pressure", "msg"])
    ap.add_argument("--n", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--fair-share", type=int, default=5)
    ap.add_argument("--capacity-per-agent", type=int, default=40)
    ap.add_argument("--parallel-runs", type=int, default=2)
    ap.add_argument("--out", default="results/traces")
    ap.add_argument("--force", action="store_true", help="re-run even if a trace exists")
    args = ap.parse_args()

    out = Path(args.out)
    jobs = []
    for n in args.n:
        for cond in args.conditions:
            for seed in args.seeds:
                path = out / f"{run_id(cond, n, seed)}.jsonl"
                if path.exists() and not args.force:
                    print(f"skip {path.name}", flush=True)
                    continue
                cfg = EnvConfig(n_agents=n, rounds=args.rounds, fair_share=args.fair_share,
                                tau=args.tau, capacity_per_agent=args.capacity_per_agent)
                jobs.append((cfg, cond, seed))
    print(f"{len(jobs)} runs to do", flush=True)
    t0 = time.time()
    agent_pool = ThreadPoolExecutor(max_workers=32)
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as ex:
        futs = {ex.submit(run_episode, cfg, cond, seed, out, agent_pool): (cfg, cond, seed)
                for cfg, cond, seed in jobs}
        done = 0
        for fu in as_completed(futs):
            cfg, cond, seed = futs[fu]
            done += 1
            try:
                p = fu.result()
                end = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
                print(f"[{done}/{len(jobs)}] {p.name} rounds={end['rounds_played']} "
                      f"collapsed={end['collapsed']} wall={end['wall_s']}s "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                print(f"[{done}/{len(jobs)}] FAILED {cond} N{cfg.n_agents} s{seed}: {e!r}",
                      flush=True)
    agent_pool.shutdown()
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
