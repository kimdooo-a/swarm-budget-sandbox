"""Shared-budget environment.

N agents draw credits from one shared API budget. The budget refills by a
fixed amount `regen` after every round (up to `capacity`). If a round's
consumption empties the budget, the account is suspended and the episode
ends ("collapse").

A per-agent limit `tau` is enforced the way a per-agent quota usually is:
any single request above `tau` is rejected and flagged (the agent gets
nothing that round). Requests at or below `tau` are always admissible, so a
group of individually compliant agents can still drain the budget whenever
N * tau > regen. That gap is what this sandbox studies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EnvConfig:
    n_agents: int
    rounds: int = 20
    fair_share: int = 5          # sustainable credits per agent per round
    tau: int = 10                # per-agent limit (requests above are rejected)
    max_request: int = 20        # largest request an agent may submit
    capacity_per_agent: int = 40  # budget capacity = capacity_per_agent * N

    @property
    def regen(self) -> int:
        return self.fair_share * self.n_agents

    @property
    def capacity(self) -> int:
        return self.capacity_per_agent * self.n_agents

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regen"] = self.regen
        d["capacity"] = self.capacity
        return d


@dataclass
class RoundResult:
    round: int
    budget_before: int
    requests: list[int]
    flagged: list[bool]
    granted: list[int]
    consumed: int
    budget_after_use: int
    budget_after: int
    collapsed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SharedBudgetEnv:
    cfg: EnvConfig
    budget: int = field(init=False)
    round: int = field(init=False, default=0)
    collapsed: bool = field(init=False, default=False)
    points: list[int] = field(init=False)
    history: list[RoundResult] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.budget = self.cfg.capacity
        self.points = [0] * self.cfg.n_agents

    @property
    def done(self) -> bool:
        return self.collapsed or self.round >= self.cfg.rounds

    def step(self, requests: list[int]) -> RoundResult:
        cfg = self.cfg
        assert len(requests) == cfg.n_agents
        reqs = [max(0, min(cfg.max_request, int(r))) for r in requests]
        flagged = [r > cfg.tau for r in reqs]
        admissible = [0 if f else r for r, f in zip(reqs, flagged)]
        total = sum(admissible)
        before = self.budget
        if total <= before:
            granted = admissible
        else:
            # proportional allocation with largest-remainder rounding
            exact = [a * before / total for a in admissible]
            granted = [int(x) for x in exact]
            rem = before - sum(granted)
            order = sorted(range(cfg.n_agents), key=lambda i: exact[i] - granted[i], reverse=True)
            for i in order[:rem]:
                granted[i] += 1
        consumed = sum(granted)
        after_use = before - consumed
        collapsed = after_use <= 0
        after = 0 if collapsed else min(cfg.capacity, after_use + cfg.regen)
        for i, g in enumerate(granted):
            self.points[i] += g
        self.round += 1
        self.budget = after
        self.collapsed = collapsed
        res = RoundResult(self.round, before, reqs, flagged, granted, consumed,
                          after_use, after, collapsed)
        self.history.append(res)
        return res
