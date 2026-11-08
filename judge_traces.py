"""Post-hoc LLM monitors: read finished JSONL traces and write judge traces.

For every round t of every run, the per-agent judge (A_llm) is called once
per agent with only that agent's window, and the swarm judge (B_llm) is
called once with the whole group's window. Both see rounds up to t only.

  python judge_traces.py --traces results/traces --out results/judges
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sandbox.monitors import judge_agent, judge_swarm
from sandbox.runner import load_trace


def judge_run(path: Path, out_dir: Path, pool: ThreadPoolExecutor, monitors: list[str]) -> Path:
    run = load_trace(path)
    start = run["start"]
    env, names, seed = start["env"], start["agents"], start["seed"]
    rounds = run["rounds"]
    with_msgs = start["condition"] == "msg"
    out = out_dir / path.name
    tmp = out.with_suffix(".jsonl.part")
    jobs = []
    for t in range(len(rounds)):
        if "A_llm" in monitors:
            for i in range(len(names)):
                jobs.append(("A_llm", t, i, pool.submit(judge_agent, env, rounds, t, i, seed)))
        if "B_llm" in monitors:
            jobs.append(("B_llm", t, None, pool.submit(judge_swarm, env, rounds, t, names, seed,
                                                       with_msgs)))
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "judge_start", "run_id": start["run_id"],
                            "monitors": monitors}) + "\n")
        for mon, t, i, fu in jobs:
            alarm, pattern, rec = fu.result()
            rec.update({"type": "judge_call", "monitor": mon, "round": rounds[t]["round"],
                        "agent": names[i] if i is not None else None})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--out", default="results/judges")
    ap.add_argument("--monitors", nargs="+", default=["A_llm", "B_llm"])
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.traces).glob(args.glob))
    pool = ThreadPoolExecutor(max_workers=16)
    t0 = time.time()
    for k, p in enumerate(paths, 1):
        if (out_dir / p.name).exists() and not args.force:
            print(f"skip {p.name}", flush=True)
            continue
        judge_run(p, out_dir, pool, args.monitors)
        print(f"[{k}/{len(paths)}] judged {p.name} elapsed={time.time() - t0:.0f}s", flush=True)
    pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
