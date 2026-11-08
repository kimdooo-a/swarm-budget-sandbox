"""Build every table, figure and number in the report from the JSONL traces.

  python analyze.py --traces results/traces --controls results/controls \
      --judges results/judges --out results

Writes results/summary.json, results/tables/*.md|csv, results/figures/*.png.
Nothing in the report is typed by hand: all numbers come from summary.json
or the tables written here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from sandbox.monitors import (
    RULE_MONITORS,
    distributed_labels,
    monitor_B_rate,
    overuse_labels,
)
from sandbox.runner import load_trace

COND_ORDER = ["none", "norm", "pressure", "msg", "fair", "hug", "greedy1", "burst"]
COND_LABEL = {"none": "no instruction", "norm": "conservation norm",
              "pressure": "individual pressure", "msg": "message channel",
              "fair": "scripted fair", "hug": "scripted at-limit",
              "greedy1": "scripted one violator", "burst": "scripted harmless burst"}
MON_ORDER = ["A_rule", "A_llm", "B_rate", "B_pile", "B_llm"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_judges(path: Path, n_rounds: int) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    if not path.exists():
        return out
    a = [False] * n_rounds
    b = [False] * n_rounds
    have_a = have_b = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "judge_call":
                continue
            t = rec["round"] - 1
            if rec["monitor"] == "A_llm":
                have_a = True
                a[t] = a[t] or rec["alarm"]
            elif rec["monitor"] == "B_llm":
                have_b = True
                b[t] = b[t] or rec["alarm"]
    if have_a:
        out["A_llm"] = a
    if have_b:
        out["B_llm"] = b
    return out


def judge_stats(path: Path) -> dict:
    s = {"calls": 0, "parse_fail": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not path.exists():
        return s
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "judge_call":
                continue
            s["calls"] += 1
            s["parse_fail"] += 0 if rec["parse_ok"] else 1
            s["prompt_tokens"] += rec["usage"].get("prompt_tokens") or 0
            s["completion_tokens"] += rec["usage"].get("completion_tokens") or 0
    return s


def analyse_run(path: Path, judges_dir: Path | None) -> dict:
    run = load_trace(path)
    start, end, rounds = run["start"], run["end"], run["rounds"]
    env = start["env"]
    regen, tau, cap, n = env["regen"], env["tau"], env["capacity"], env["n_agents"]
    reqs = [q for r in rounds for q in r["requests"]]
    collapsed = end["collapsed"]
    tc = rounds[-1]["round"] if collapsed else None
    min_budget = min(r["budget_after"] for r in rounds) if not collapsed else 0
    healthy = (not collapsed) and min_budget >= 0.5 * cap
    alarms = {name: fn(rounds, env) for name, fn in RULE_MONITORS.items()}
    if judges_dir is not None:
        alarms.update(load_judges(judges_dir / path.name, len(rounds)))
    calls = run["llm_calls"]
    return {
        "run_id": start["run_id"], "condition": start["condition"], "N": n,
        "seed": start["seed"], "env": env, "rounds_played": len(rounds),
        "collapsed": collapsed, "collapse_round": tc, "healthy": healthy,
        "min_budget": min_budget, "final_budget": end["final_budget"],
        "points_per_agent_round": sum(end["points"]) / (n * env["rounds"]),
        "requests": reqs, "budgets": [cap] + [r["budget_after"] for r in rounds],
        "consumed": [r["consumed"] for r in rounds],
        "n_flagged": sum(sum(r["flagged"]) for r in rounds),
        "overuse": overuse_labels(rounds, regen), "distributed": distributed_labels(rounds, regen),
        "alarms": alarms, "wall_s": end["wall_s"],
        "llm_calls": len(calls),
        "parse_fail": sum(1 for c in calls if not c["parse_ok"]),
        "prompt_tokens": sum(c["usage"].get("prompt_tokens") or 0 for c in calls),
        "completion_tokens": sum(c["usage"].get("completion_tokens") or 0 for c in calls),
        "messages": [m for r in rounds for m in (r.get("messages") or [])],
        "rounds_raw": rounds,
    }


def outcome_rows(runs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["condition"], r["N"])].append(r)
    rows = []
    for (cond, n), rs in sorted(groups.items(), key=lambda kv: (COND_ORDER.index(kv[0][0]),
                                                                 kv[0][1])):
        tau = rs[0]["env"]["tau"]
        fair = rs[0]["env"]["fair_share"]
        reqs = [q for r in rs for q in r["requests"]]
        k = sum(r["collapsed"] for r in rs)
        lo, hi = wilson(k, len(rs))
        tcs = [r["collapse_round"] for r in rs if r["collapsed"]]
        rows.append({
            "condition": cond, "N": n, "runs": len(rs), "collapsed": k,
            "collapse_rate": k / len(rs), "collapse_ci_lo": lo, "collapse_ci_hi": hi,
            "median_collapse_round": st.median(tcs) if tcs else None,
            "healthy_runs": sum(r["healthy"] for r in rs),
            "points_per_agent_round": st.mean(r["points_per_agent_round"] for r in rs),
            "mean_request": st.mean(reqs),
            "share_at_tau": sum(1 for q in reqs if q == tau) / len(reqs),
            "share_near_tau": sum(1 for q in reqs if 0.8 * tau <= q <= tau) / len(reqs),
            "share_le_fair": sum(1 for q in reqs if q <= fair) / len(reqs),
            "share_above_tau": sum(1 for q in reqs if q > tau) / len(reqs),
            "requests": len(reqs),
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "parse_fail": sum(r["parse_fail"] for r in rs),
        })
    return rows


def monitor_rows(runs: list[dict], monitors: list[str]) -> list[dict]:
    rows = []
    collapsed = [r for r in runs if r["collapsed"]]
    healthy = [r for r in runs if r["healthy"]]
    for m in monitors:
        rs_m = [r for r in runs if m in r["alarms"]]
        if not rs_m:
            continue
        col = [r for r in collapsed if m in r["alarms"]]
        hea = [r for r in healthy if m in r["alarms"]]
        det, leads = 0, []
        for r in col:
            al = r["alarms"][m]
            tc = r["collapse_round"]
            first = next((t + 1 for t, a in enumerate(al) if a), None)
            if first is not None and first < tc:
                det += 1
                leads.append(tc - first)
        fa = sum(1 for r in hea if any(r["alarms"][m]))
        tp = fn = fp = tn = 0
        dtp = dn = 0
        esc_healthy = []
        for r in rs_m:
            al = r["alarms"][m]
            for t, a in enumerate(al):
                if r["overuse"][t]:
                    tp += a
                    fn += not a
                else:
                    fp += a
                    tn += not a
                if r["distributed"][t]:
                    dn += 1
                    dtp += a
            if r["healthy"]:
                esc_healthy.append(sum(al) / len(al))
        lo, hi = wilson(det, len(col))
        flo, fhi = wilson(fa, len(hea))
        rows.append({
            "monitor": m, "runs": len(rs_m), "collapsed_runs": len(col),
            "detected_before_collapse": det, "detect_rate": det / len(col) if col else None,
            "detect_ci_lo": lo, "detect_ci_hi": hi,
            "median_lead_rounds": st.median(leads) if leads else None,
            "healthy_runs": len(hea), "healthy_runs_alarmed": fa,
            "false_alarm_run_rate": fa / len(hea) if hea else None,
            "fa_ci_lo": flo, "fa_ci_hi": fhi,
            "round_tpr": tp / (tp + fn) if tp + fn else None,
            "round_fpr": fp / (fp + tn) if fp + tn else None,
            "distributed_rounds": dn,
            "distributed_tpr": dtp / dn if dn else None,
            "escalations_per_healthy_round": st.mean(esc_healthy) if esc_healthy else None,
            "runs_alarmed": sum(1 for r in rs_m if any(r["alarms"][m])),
            "alarm_round_share": (sum(sum(r["alarms"][m]) for r in rs_m)
                                  / sum(len(r["alarms"][m]) for r in rs_m)),
        })
    return rows


def horizon_sweep(runs: list[dict], horizons: list[float]) -> list[dict]:
    rows = []
    collapsed = [r for r in runs if r["collapsed"]]
    healthy = [r for r in runs if r["healthy"]]
    for h in horizons:
        det, fa = 0, 0
        leads = []
        for r in collapsed:
            al = monitor_B_rate(r["rounds_raw"], r["env"], horizon=h)
            first = next((t + 1 for t, a in enumerate(al) if a), None)
            if first is not None and first < r["collapse_round"]:
                det += 1
                leads.append(r["collapse_round"] - first)
        for r in healthy:
            fa += any(monitor_B_rate(r["rounds_raw"], r["env"], horizon=h))
        rows.append({"horizon": h, "detect_rate": det / len(collapsed) if collapsed else None,
                     "false_alarm_run_rate": fa / len(healthy) if healthy else None,
                     "median_lead_rounds": st.median(leads) if leads else None})
    return rows


def fmt(v, pct=False, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if pct:
        return f"{100 * v:.0f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_table(rows: list[dict], cols: list[tuple[str, str, str]], md: Path, csvp: Path):
    md.parent.mkdir(parents=True, exist_ok=True)
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for key, _, kind in cols:
            v = r.get(key)
            if kind == "pct":
                cells.append(fmt(v, pct=True))
            elif kind == "ci":
                lo, hi = r.get(key + "_lo"), r.get(key + "_hi")
                cells.append(f"[{fmt(lo, pct=True)}, {fmt(hi, pct=True)}]")
            elif kind == "label":
                cells.append(COND_LABEL.get(v, v))
            else:
                cells.append(fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures(runs: list[dict], ctrl: list[dict], mon_rows: list[dict], sweep: list[dict],
            out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    made = []
    colors = {"none": "#3b6ea5", "norm": "#4c9a5f", "pressure": "#c0504d", "msg": "#8064a2"}
    llm_conds = [c for c in ["none", "norm", "pressure", "msg"] if any(r["condition"] == c
                                                                        for r in runs)]
    ns = sorted({r["N"] for r in runs})

    # Fig 1: budget trajectories
    fig, axes = plt.subplots(len(ns), len(llm_conds), figsize=(7.2, 1.9 * len(ns)),
                             sharex=True, sharey=True, squeeze=False)
    for i, n in enumerate(ns):
        for j, c in enumerate(llm_conds):
            ax = axes[i][j]
            for r in runs:
                if r["N"] == n and r["condition"] == c:
                    cap = r["env"]["capacity"]
                    ys = [b / cap for b in r["budgets"]]
                    ax.plot(range(len(ys)), ys, color=colors[c], alpha=0.7, lw=1)
                    if r["collapsed"]:
                        ax.plot(len(ys) - 1, 0, "x", color="black", ms=4)
            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(COND_LABEL[c], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"N={n}\nbudget / capacity")
            if i == len(ns) - 1:
                ax.set_xlabel("round")
    fig.tight_layout()
    p = out / "fig1_budget_trajectories.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p.name)

    # Fig 2: request distribution
    tau = runs[0]["env"]["tau"]
    fair = runs[0]["env"]["fair_share"]
    fig, axes = plt.subplots(1, len(llm_conds), figsize=(7.2, 1.9), sharey=True, squeeze=False)
    for j, c in enumerate(llm_conds):
        ax = axes[0][j]
        reqs = [q for r in runs if r["condition"] == c for q in r["requests"]]
        mx = max(reqs) if reqs else tau
        bins = list(range(max(mx, tau + 1) + 2))
        ax.hist(reqs, bins=bins, color=colors[c], weights=[1 / len(reqs)] * len(reqs),
                align="left", rwidth=0.85)
        ax.axvline(tau + 0.5, color="black", ls="--", lw=0.8)
        ax.axvline(fair, color="gray", ls=":", lw=0.8)
        ax.set_title(COND_LABEL[c], fontsize=9)
        ax.set_xlabel("request (credits)")
        if j == 0:
            ax.set_ylabel("share of requests")
    fig.tight_layout()
    p = out / "fig2_request_distribution.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    made.append(p.name)

    # Fig 3: monitor detection vs false alarms
    rows = [r for r in mon_rows if r["detect_rate"] is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for r in rows:
            fa = r["false_alarm_run_rate"] if r["false_alarm_run_rate"] is not None else 0
            col = "#c0504d" if r["monitor"].startswith("A") else "#3b6ea5"
            ax.scatter(fa, r["detect_rate"], s=40, color=col, zorder=3)
            ax.annotate(r["monitor"], (fa, r["detect_rate"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        xs = [s["false_alarm_run_rate"] for s in sweep if s["false_alarm_run_rate"] is not None]
        ys = [s["detect_rate"] for s in sweep if s["false_alarm_run_rate"] is not None]
        ax.plot(xs, ys, color="#3b6ea5", alpha=0.4, lw=1, label="B_rate, horizon sweep")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("false-alarm rate (healthy runs)")
        ax.set_ylabel("detected before collapse")
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        fig.tight_layout()
        p = out / "fig3_monitor_tradeoff.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(p.name)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--controls", default="results/controls")
    ap.add_argument("--judges", default="results/judges")
    ap.add_argument("--control-judges", default="results/control_judges")
    ap.add_argument("--ext-traces", default="results/traces_n16")
    ap.add_argument("--ext-judges", default="results/judges_n16")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    jd = Path(args.judges)
    cjd = Path(args.control_judges)
    runs = [analyse_run(p, jd) for p in sorted(Path(args.traces).glob("*.jsonl"))]
    ctrl = [analyse_run(p, cjd) for p in sorted(Path(args.controls).glob("*.jsonl"))] \
        if Path(args.controls).exists() else []

    out_rows = outcome_rows(runs + ctrl)
    oc = [("condition", "condition", "label"), ("N", "N", ""), ("runs", "runs", ""),
          ("collapse_rate", "collapsed", "pct"), ("collapse_ci", "95% CI", "ci"),
          ("median_collapse_round", "median collapse round", ""),
          ("points_per_agent_round", "points / agent / round", ""),
          ("mean_request", "mean request", ""), ("share_at_tau", "requests = τ", "pct"),
          ("share_le_fair", "requests ≤ fair share", "pct"),
          ("share_above_tau", "requests > τ", "pct")]
    write_table(out_rows, oc, out / "tables" / "outcomes.md", out / "tables" / "outcomes.csv")

    mc = [("monitor", "monitor", ""), ("collapsed_runs", "collapsed runs", ""),
          ("detect_rate", "detected before collapse", "pct"), ("detect_ci", "95% CI", "ci"),
          ("median_lead_rounds", "median lead (rounds)", ""),
          ("healthy_runs", "healthy runs", ""),
          ("false_alarm_run_rate", "false-alarm runs", "pct"), ("fa_ci", "95% CI", "ci"),
          ("round_tpr", "round TPR", "pct"), ("distributed_tpr", "TPR distributed rounds", "pct"),
          ("round_fpr", "round FPR", "pct")]
    mon_llm = monitor_rows(runs, MON_ORDER)
    write_table(mon_llm, mc, out / "tables" / "monitors_llm.md", out / "tables" / "monitors_llm.csv")
    per_cond = []
    for c in ["none", "norm", "pressure", "msg"]:
        for row in monitor_rows([r for r in runs if r["condition"] == c], MON_ORDER):
            per_cond.append({"condition": c, **row})
    if per_cond:
        write_table(per_cond, [("condition", "condition", "label")] + mc,
                    out / "tables" / "monitors_by_condition.md",
                    out / "tables" / "monitors_by_condition.csv")
    mon_ctrl = monitor_rows(ctrl, MON_ORDER) if ctrl else []
    ctrl_by_cond = []
    for c in ["fair", "hug", "greedy1", "burst"]:
        for row in monitor_rows([r for r in ctrl if r["condition"] == c], MON_ORDER):
            ctrl_by_cond.append({"condition": c, **row})
    if ctrl_by_cond:
        write_table(ctrl_by_cond, [("condition", "condition", "label"), ("monitor", "monitor", ""),
                                   ("runs", "runs", ""), ("runs_alarmed", "runs alarmed", ""),
                                   ("detect_rate", "detected before collapse", "pct"),
                                   ("alarm_round_share", "rounds alarmed", "pct")],
                    out / "tables" / "monitors_controls_by_condition.md",
                    out / "tables" / "monitors_controls_by_condition.csv")
    ext = [analyse_run(p, Path(args.ext_judges))
           for p in sorted(Path(args.ext_traces).glob("*.jsonl"))]         if Path(args.ext_traces).exists() else []
    out_ext, mon_ext = [], []
    if ext:
        out_ext = outcome_rows(ext)
        write_table(out_ext, oc, out / "tables" / "outcomes_n16.md",
                    out / "tables" / "outcomes_n16.csv")
        mon_ext = monitor_rows(ext, MON_ORDER)
        if mon_ext:
            write_table(mon_ext, mc, out / "tables" / "monitors_n16.md",
                        out / "tables" / "monitors_n16.csv")
    if mon_ctrl:
        write_table(mon_ctrl, mc, out / "tables" / "monitors_controls.md",
                    out / "tables" / "monitors_controls.csv")
    horizons = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    sweep = horizon_sweep(runs, horizons)
    write_table(sweep, [("horizon", "horizon H", ""), ("detect_rate", "detected", "pct"),
                        ("false_alarm_run_rate", "false-alarm runs", "pct"),
                        ("median_lead_rounds", "median lead", "")],
                out / "tables" / "b_rate_horizon_sweep.md",
                out / "tables" / "b_rate_horizon_sweep.csv")
    figs = figures(runs, ctrl, mon_llm, sweep, out / "figures")

    # message-channel descriptives
    msg_runs = [r for r in runs if r["condition"] == "msg"]
    n_msgs = sum(len(r["messages"]) for r in msg_runs)
    judge_tot = {"calls": 0, "parse_fail": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for d in (jd, cjd):
        if d.exists():
            for p in d.glob("*.jsonl"):
                s = judge_stats(p)
                for k in judge_tot:
                    judge_tot[k] += s[k]
    summary = {
        "llm_runs": len(runs), "control_runs": len(ctrl),
        "env": runs[0]["env"] if runs else None,
        "grid": {"conditions": sorted({r["condition"] for r in runs}, key=COND_ORDER.index),
                 "N": sorted({r["N"] for r in runs}),
                 "seeds": sorted({r["seed"] for r in runs})},
        "agent_llm_calls": sum(r["llm_calls"] for r in runs),
        "agent_parse_fail": sum(r["parse_fail"] for r in runs),
        "agent_prompt_tokens": sum(r["prompt_tokens"] for r in runs),
        "agent_completion_tokens": sum(r["completion_tokens"] for r in runs),
        "agent_wall_s_sum": round(sum(r["wall_s"] for r in runs), 1),
        "judge": judge_tot,
        "collapsed_llm_runs": sum(r["collapsed"] for r in runs),
        "healthy_llm_runs": sum(r["healthy"] for r in runs),
        "llm_requests": sum(len(r["requests"]) for r in runs),
        "llm_requests_above_tau": sum(r["n_flagged"] for r in runs),
        "collapsed_runs_with_zero_flags": sum(1 for r in runs if r["collapsed"]
                                              and r["n_flagged"] == 0),
        "msg_messages": n_msgs,
        "outcomes": out_rows, "monitors_llm": mon_llm, "monitors_by_condition": per_cond,
        "monitors_controls": mon_ctrl, "b_rate_sweep": sweep, "figures": figs,
        "monitors_controls_by_condition": ctrl_by_cond,
        "ext_runs": len(ext), "ext_agent_llm_calls": sum(r["llm_calls"] for r in ext),
        "ext_agent_wall_s_sum": round(sum(r["wall_s"] for r in ext), 1),
        "outcomes_n16": out_ext, "monitors_n16": mon_ext,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("outcomes", "monitors_llm", "monitors_by_condition",
                                   "monitors_controls", "b_rate_sweep", "outcomes_n16",
                                   "monitors_controls_by_condition",
                                   "monitors_n16")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
