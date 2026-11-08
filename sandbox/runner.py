"""Run one episode and write a complete JSONL trace."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agents import (
    LLM_CONDITIONS,
    SCRIPTED_CONDITIONS,
    agent_names,
    llm_decide,
    scripted_decide,
)
from .env import EnvConfig, SharedBudgetEnv


def run_id(condition: str, n: int, seed: int) -> str:
    return f"{condition}_N{n}_s{seed}"


def run_episode(cfg: EnvConfig, condition: str, seed: int, out_dir: Path,
                pool: ThreadPoolExecutor | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id(condition, cfg.n_agents, seed)
    path = out_dir / f"{rid}.jsonl"
    tmp = path.with_suffix(".jsonl.part")
    env = SharedBudgetEnv(cfg)
    names = agent_names(cfg.n_agents)
    rng = random.Random(seed)
    messages: list[tuple[str, str]] | None = [] if condition == "msg" else None
    t_start = time.time()
    own_pool = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=cfg.n_agents)
    with tmp.open("w", encoding="utf-8") as f:
        def w(rec: dict) -> None:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        w({"type": "run_start", "run_id": rid, "condition": condition, "seed": seed,
           "env": cfg.to_dict(), "agents": names, "time": t_start})
        while not env.done:
            t = env.round + 1
            budget = env.budget
            if condition in LLM_CONDITIONS:
                futs = [pool.submit(llm_decide, cfg, condition, seed, i, names[i], t, budget,
                                    list(env.history), messages) for i in range(cfg.n_agents)]
                outs = [fu.result() for fu in futs]
                requests = [o[0] for o in outs]
                for o in outs:
                    w(o[2])
                if condition == "msg":
                    messages = [(names[i], outs[i][1]) for i in range(cfg.n_agents)
                                if outs[i][1]]
            elif condition in SCRIPTED_CONDITIONS:
                requests = [scripted_decide(cfg, condition, rng, i, t) for i in range(cfg.n_agents)]
            else:
                raise ValueError(condition)
            res = env.step(requests)
            rec = {"type": "round", **res.to_dict()}
            if condition == "msg":
                rec["messages"] = messages
            w(rec)
        w({"type": "run_end", "run_id": rid, "rounds_played": env.round,
           "collapsed": env.collapsed, "final_budget": env.budget, "points": env.points,
           "wall_s": round(time.time() - t_start, 2)})
    tmp.replace(path)
    if own_pool:
        pool.shutdown()
    return path


def load_trace(path: Path) -> dict:
    run = {"llm_calls": [], "rounds": []}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ty = rec["type"]
            if ty == "run_start":
                run["start"] = rec
            elif ty == "run_end":
                run["end"] = rec
            elif ty == "round":
                run["rounds"].append(rec)
            elif ty == "llm_call":
                run["llm_calls"].append(rec)
    return run
