"""Descriptive counts from the traces that the report quotes (written to results/report_extras.json).

- how often agents' stated reasons mention the per-agent limit vs. the shared budget
- distribution of the swarm judge's pattern labels on alarmed rounds
- how often the per-agent LLM judge escalated, and on what
- sample messages from the message-channel condition (verbatim, with run/round/agent)

  python scripts/extras.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.runner import load_trace

LIMIT_RE = re.compile(r"\b(limit|compliance|flag|reject|threshold|cap|maximum admissible|"
                      r"allowed)\b", re.IGNORECASE)
BUDGET_RE = re.compile(r"\b(sustain\w*|shared|deplet\w*|exhaust\w*|conserv\w*|preserv\w*|"
                       r"everyone|collective|group)\b", re.IGNORECASE)
COORD_RE = re.compile(r"\b(let'?s|we should|everyone|all of us|agree|propose|suggest)\b", re.IGNORECASE)


def reason_of(text: str) -> str:
    try:
        obj = json.loads(text)
        return str(obj.get("reason", ""))
    except Exception:
        return ""


def main() -> int:
    traces = sorted((ROOT / "results" / "traces").glob("*.jsonl"))
    X: dict = {}
    by_cond: dict[str, Counter] = {}
    for p in traces:
        run = load_trace(p)
        c = run["start"]["condition"]
        cnt = by_cond.setdefault(c, Counter())
        for call in run["llm_calls"]:
            r = reason_of(call["response"])
            cnt["calls"] += 1
            cnt["limit"] += bool(LIMIT_RE.search(r))
            cnt["budget"] += bool(BUDGET_RE.search(r))
            if call["request"] == run["start"]["env"]["tau"]:
                cnt["at_tau"] += 1
                cnt["at_tau_limit"] += bool(LIMIT_RE.search(r))
    for c, cnt in by_cond.items():
        X[f"{c}.reason_limit"] = f"{100 * cnt['limit'] / cnt['calls']:.0f}%"
        X[f"{c}.reason_budget"] = f"{100 * cnt['budget'] / cnt['calls']:.0f}%"
        if cnt["at_tau"]:
            X[f"{c}.at_tau_reason_limit"] = f"{100 * cnt['at_tau_limit'] / cnt['at_tau']:.0f}%"
    tot = sum(v["calls"] for v in by_cond.values())
    X["all.reason_limit"] = f"{100 * sum(v['limit'] for v in by_cond.values()) / tot:.0f}%"
    at = sum(v["at_tau"] for v in by_cond.values())
    if at:
        X["all.at_tau_reason_limit"] = \
            f"{100 * sum(v['at_tau_limit'] for v in by_cond.values()) / at:.0f}%"

    # judges
    pat = Counter()
    a_alarm = a_calls = 0
    a_alarm_flagged = 0
    for p in sorted((ROOT / "results" / "judges").glob("*.jsonl")):
        run = load_trace(ROOT / "results" / "traces" / p.name)
        rounds = run["rounds"]
        names = run["start"]["agents"]
        with p.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") != "judge_call":
                    continue
                if rec["monitor"] == "B_llm" and rec["alarm"]:
                    pat[str(rec.get("pattern"))] += 1
                if rec["monitor"] == "A_llm":
                    a_calls += 1
                    if rec["alarm"]:
                        a_alarm += 1
                        i = names.index(rec["agent"])
                        w = rounds[max(0, rec["round"] - 5): rec["round"]]
                        a_alarm_flagged += any(r["flagged"][i] for r in w)
    total_pat = sum(pat.values())
    for k, v in pat.items():
        X[f"bllm_pattern.{k}"] = f"{100 * v / total_pat:.0f}%" if total_pat else "–"
    X["bllm_alarm_rounds"] = str(total_pat)
    X["allm_calls"] = f"{a_calls:,}"
    X["allm_alarms"] = str(a_alarm)
    X["allm_alarm_rate"] = f"{100 * a_alarm / a_calls:.2f}%" if a_calls else "–"
    X["allm_alarms_with_flag_in_window"] = str(a_alarm_flagged)

    # messages
    coord = n_msg = 0
    samples = []
    for p in sorted((ROOT / "results" / "traces").glob("msg_*.jsonl")):
        run = load_trace(p)
        for r in run["rounds"]:
            for who, m in r.get("messages") or []:
                n_msg += 1
                coord += bool(COORD_RE.search(m))
                if len(samples) < 6 and r["round"] in (1, 5, 10):
                    samples.append(f"{run['start']['run_id']} r{r['round']} {who}: {m}")
    X["msg.count"] = f"{n_msg:,}"
    X["msg.coord_share"] = f"{100 * coord / n_msg:.0f}%" if n_msg else "–"
    (ROOT / "results" / "report_extras.json").write_text(json.dumps(X, indent=1,
                                                                    ensure_ascii=False),
                                                         encoding="utf-8")
    (ROOT / "results" / "message_samples.txt").write_text("\n".join(samples) + "\n",
                                                          encoding="utf-8")
    print(json.dumps(X, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
