"""Compare agents on one declared problem set.

Harness ``accuracy_score`` is resolved/submitted for whatever the report
happened to contain. The cached initial Polyglot run is the full suite
(about 225 problems). A staged live verify starts on ``small.json`` and,
when that screen clears DGM's threshold, also runs ``medium.json``.
Those fractions are not comparable: the same initial agent is 0.142 on
the full suite and 0.20 on the small subset. Accept uses the count of
resolved ids on the screen every candidate finishes, and only when both
agents have a result for every problem in that set.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

from evopilot.types import AgentSnapshot, VerificationEvidence


class VerifySchedule(NamedTuple):
    """DGM's Polyglot screen, then the larger subset when the screen clears.

    ``score_protocol`` is the screen. Every candidate is evaluated there, so
    accept and rollback stay on one denominator. ``test_task_list_more`` is
    the second stage DGM stores as ``overall_performance_deep``.
    """

    test_task_list: tuple[str, ...]
    test_more_threshold: float | None
    test_task_list_more: tuple[str, ...] | None
    score_protocol: tuple[str, ...]


def verify_schedule(small: list, medium: list, *, shallow: bool, threshold: float = 0.4) -> VerifySchedule:
    """Match ``DGM_outer`` staging.

    The default runs ``small``, then ``medium`` when
    ``total_resolved_instances >= threshold * len(small)``. ``shallow``
    keeps the screen and skips the second stage.
    """

    screen = tuple(small)
    if not screen:
        raise RuntimeError("The evaluation screen is empty.")
    if shallow:
        return VerifySchedule(screen, None, None, screen)
    if threshold < 0:
        raise RuntimeError("The staging threshold must be non-negative.")
    return VerifySchedule(screen, float(threshold), tuple(medium), screen)


def evaluation_protocol(small: list, medium: list, *, shallow: bool) -> tuple[str, ...]:
    """One combined problem set.

    ``shallow`` is ``small`` alone. Otherwise this is the unique union of
    ``small`` and ``medium``. The live loop does not use the union: staged
    verify calls ``verify_schedule`` and scores the screen.
    """

    chosen = list(small) if shallow else list(small) + list(medium)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in chosen:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    if not ordered:
        raise RuntimeError("The evaluation protocol is empty.")
    return tuple(ordered)


def partition(
    resolved,
    unresolved,
    empty,
    protocol: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split ``protocol`` into resolved, unresolved, and empty.

    An id that appears in more than one list counts once, preferring
    resolved, then unresolved, then empty. Ids outside the protocol are
    dropped. The last tuple is the protocol ids with no result.
    """

    resolved_set = set(resolved)
    unresolved_set = set(unresolved)
    empty_set = set(empty)
    kept_resolved: list[str] = []
    kept_unresolved: list[str] = []
    kept_empty: list[str] = []
    missing: list[str] = []
    for item in protocol:
        if item in resolved_set:
            kept_resolved.append(item)
        elif item in unresolved_set:
            kept_unresolved.append(item)
        elif item in empty_set:
            kept_empty.append(item)
        else:
            missing.append(item)
    return tuple(kept_resolved), tuple(kept_unresolved), tuple(kept_empty), tuple(missing)


def align_verification(evidence: VerificationEvidence, origin: AgentSnapshot) -> VerificationEvidence:
    """Replace raw fractions with protocol accuracies.

    A missing result on either side leaves ``candidate_score`` unset and
    marks ``protocol_mismatch`` so the policy cannot accept the candidate.
    Structural failures keep their own cluster.
    """

    protocol = evidence.score_protocol
    resolved, unresolved, empty, missing = partition(
        evidence.resolved_ids,
        evidence.unresolved_ids,
        evidence.empty_patch_ids,
        protocol,
    )
    base_resolved, _, _, base_missing = partition(
        origin.resolved_ids,
        origin.unresolved_ids,
        origin.empty_patch_ids,
        protocol,
    )
    if missing or base_missing:
        if evidence.structural_error:
            return replace(evidence, base_agent_id=origin.agent_id, base_score=origin.score)
        return replace(
            evidence,
            base_agent_id=origin.agent_id,
            base_score=origin.score,
            candidate_score=None,
            protocol_mismatch=True,
            regression=False,
            local_error=False,
            failure_cluster="incomparable_protocol",
            resolved_ids=resolved,
            unresolved_ids=unresolved,
            empty_patch_ids=empty,
        )

    base_score = len(base_resolved) / len(protocol)
    candidate_score = len(resolved) / len(protocol)
    lost = [item for item in base_resolved if item in unresolved or item in empty]
    regression = bool(lost) or candidate_score < base_score - 1e-12
    cluster = evidence.failure_cluster
    if lost and not str(cluster).startswith("regression") and cluster in {"ok", "no_gain", "incomparable_protocol"}:
        cluster = "regression"
    elif (
        not lost
        and candidate_score <= base_score + 1e-12
        and cluster in {"ok", "incomparable_protocol"}
    ):
        cluster = "no_gain"
    elif (
        not lost
        and candidate_score > base_score + 1e-12
        and cluster in {"no_gain", "regression", "incomparable_protocol"}
    ):
        cluster = "ok"
    return replace(
        evidence,
        base_agent_id=origin.agent_id,
        base_score=base_score,
        candidate_score=candidate_score,
        protocol_mismatch=False,
        regression=regression,
        resolved_ids=resolved,
        unresolved_ids=unresolved,
        empty_patch_ids=empty,
        observed_regression=",".join(lost) if lost else None,
        failure_cluster=cluster,
    )
