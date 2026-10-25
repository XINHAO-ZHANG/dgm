"""Stagnation is not only a flat best score.

Over a recent window the detector combines four signals from the design:

- little or no best-so-far gain
- low intent novelty
- a high rollback rate
- a repeated failure cluster, including improve → debug → rollback loops
"""

from __future__ import annotations

from collections import Counter

from evopilot.types import AttemptOutcome, StagnationSignal, no_stagnation


def detect_stagnation(
    attempts: list[AttemptOutcome],
    *,
    window: int = 4,
    min_steps: int = 3,
) -> StagnationSignal:
    if window < 1:
        raise ValueError("stagnation window must be >= 1")
    recent = list(attempts[-window:])
    if len(recent) < min_steps:
        return no_stagnation()

    delta_q = recent[-1].best_after - recent[0].best_before
    intents = [item.intent for item in recent if item.intent]
    novelty = (len(set(intents)) / len(recent)) if recent else 1.0
    rollbacks = sum(1 for item in recent if item.decision == "rollback")
    rollback_rate = rollbacks / len(recent)
    rollback_loop = sum(
        1 for item in recent if item.decision == "rollback" and item.debug_count > 0
    ) >= 2

    clusters = [item.failure_cluster for item in recent if item.failure_cluster not in {"", "ok", "none"}]
    dominant = None
    fraction = 0.0
    if clusters:
        dominant, count = Counter(clusters).most_common(1)[0]
        fraction = count / len(recent)

    reasons: list[str] = []
    if delta_q <= 1e-12:
        reasons.append("no_best_score_gain")
    if novelty <= 0.5:
        reasons.append("low_intent_novelty")
    if rollback_rate >= 0.5:
        reasons.append("high_rollback_rate")
    if fraction >= 0.6:
        reasons.append("repeated_failure_cluster")
    if rollback_loop:
        reasons.append("improve_debug_rollback_loop")

    stagnant = delta_q <= 1e-12 and (
        rollback_rate >= 0.5
        or fraction >= 0.6
        or (novelty <= 0.5 and rollback_loop)
    )
    return StagnationSignal(
        stagnant=stagnant,
        reasons=tuple(reasons),
        delta_q=delta_q,
        novelty=novelty,
        rollback_rate=rollback_rate,
        dominant_cluster=dominant,
        dominant_cluster_fraction=fraction,
        rollback_loop=rollback_loop,
    )
