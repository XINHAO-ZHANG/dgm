"""Compute ledger for AUEC-style curves.

Q_best(c) is the best *accepted* score once cumulative evaluations reach c.
Q_active(c) is the score of the agent the controller is currently evolving.
Rejected candidates do not raise Q_best.

The integral is the area of the step curve::

    AUEC(C) = sum_i Q(c_i) * (c_{i+1} - c_i)

Q(c_i) is the quality *after* the event at c_i and is held until the next
event. A discovery at the final budget adds no further area until more
compute is spent. ``normalized_auec`` divides by C when C > 0.

Evaluations are the primary compute proxy because that is the unit this
repo already spends on Polyglot and SWE-bench. Tokens are recorded only
when a caller supplies them. Missing token totals stay null; they are not
estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComputeLedger:
    events: list[dict] = field(default_factory=list)
    cumulative_evaluations: int = 0
    cumulative_tokens: int | None = None

    def record(
        self,
        *,
        event: str,
        step: int,
        active_score: float,
        best_score: float,
        candidate_score: float | None,
        decision: str | None,
        agent_id: str,
        evaluations: int = 0,
        tokens: int | None = None,
        details: dict | None = None,
    ) -> dict:
        spent = max(0, int(evaluations))
        self.cumulative_evaluations += spent
        if tokens is not None:
            self.cumulative_tokens = (self.cumulative_tokens or 0) + int(tokens)
        row = {
            "step": step,
            "event": event,
            "decision": decision,
            "agent_id": agent_id,
            "candidate_score": candidate_score,
            "active_score": active_score,
            "best_score": best_score,
            "evaluations": spent,
            "tokens": tokens,
            "cumulative_evaluations": self.cumulative_evaluations,
            "cumulative_tokens": self.cumulative_tokens,
            "details": details or {},
        }
        self.events.append(row)
        return row

    def area(self, key: str) -> float:
        total = 0.0
        events = self.events
        for earlier, later in zip(events, events[1:]):
            width = later["cumulative_evaluations"] - earlier["cumulative_evaluations"]
            if width > 0:
                total += float(earlier[key]) * width
        return total

    def auec(self) -> dict:
        budget = float(self.cumulative_evaluations)
        best = self.area("best_score")
        active = self.area("active_score")
        return {
            "auec_best": best,
            "auec_active": active,
            "normalized_auec_best": (best / budget) if budget else 0.0,
            "normalized_auec_active": (active / budget) if budget else 0.0,
            "budget_evaluations": budget,
            "curve": [
                {
                    "cumulative_evaluations": event["cumulative_evaluations"],
                    "best_score": event["best_score"],
                    "active_score": event["active_score"],
                    "event": event["event"],
                }
                for event in self.events
            ],
        }
