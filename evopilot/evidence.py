"""Build controller evidence from a DGM ``metadata.json`` payload.

Score, resolved ids, and submitted-instance counts come from the harness
report already stored by ``self_improve``. Token totals are copied only
when the metadata actually contains them.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import replace

from evopilot.scoring import partition
from evopilot.types import AgentSnapshot, VerificationEvidence


def parse_diff_paths(diff_text: str) -> tuple[str, ...]:
    files: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[-1]
        if path.startswith("b/"):
            path = path[2:]
        if path not in files:
            files.append(path)
    return tuple(files)


def failure_cluster_for(
    *,
    compiled: bool,
    structural: bool,
    lost: list[str],
    resolved_target: bool | None,
    entry: str | None,
    score: float | None,
    base_score: float,
    override: str | None = None,
) -> str:
    if override:
        return override
    if not compiled:
        return "compile_or_empty_patch"
    if lost:
        languages = sorted({item.split("__", 1)[0] for item in lost if "__" in item})
        return "regression:" + ",".join(languages) if languages else "regression"
    if structural:
        return "catastrophic_drop"
    if resolved_target is False and entry:
        return f"missed:{entry}"
    if score is not None and score <= base_score + 1e-12:
        return "no_gain"
    return "ok"


def extract_tokens(metadata: dict) -> int | None:
    direct = metadata.get("total_tokens")
    if isinstance(direct, int):
        return direct
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    for key in ("total_tokens", "total"):
        if isinstance(usage.get(key), int):
            return usage[key]
    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    if isinstance(prompt, int) or isinstance(completion, int):
        return int(prompt or 0) + int(completion or 0)
    return None


def variance_from_result_files(results_dir: str | None, run_keyword: str | None) -> float | None:
    """Sample variance across harness report files for one candidate.

    DGM stores one JSON report per repeated evaluation. A single report has
    no variance, so escalate stays unavailable until at least two reports exist.
    """

    if not results_dir or not run_keyword or not os.path.isdir(results_dir):
        return None
    scores: list[float] = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json") or run_keyword not in name:
            continue
        path = os.path.join(results_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        submitted = payload.get("submitted_instances")
        resolved = payload.get("resolved_instances")
        if not isinstance(submitted, int) or submitted <= 0 or not isinstance(resolved, int):
            continue
        scores.append(resolved / submitted)
    if len(scores) < 2:
        return None
    return float(statistics.pstdev(scores))


def extract_variance(metadata: dict) -> float | None:
    scores = metadata.get("per_eval_scores")
    if not isinstance(scores, list) or len(scores) < 2:
        return None
    try:
        values = [float(score) for score in scores]
    except (TypeError, ValueError):
        return None
    return float(statistics.pstdev(values))


def evidence_from_dgm_metadata(
    metadata: dict,
    *,
    base: AgentSnapshot,
    debug_round: int = 0,
    escalations_used: int = 0,
    lineage_id: str = "",
    patch_path: str | None = None,
    results_dir: str | None = None,
    already_counted_evaluations: int = 0,
    catastrophic_drop: float = 0.2,
    change_intent: str = "",
    protocol_ids: tuple[str, ...] | None = None,
) -> VerificationEvidence:
    perf = metadata.get("overall_performance") or {}
    deep = metadata.get("overall_performance_deep") or {}
    if not isinstance(perf, dict):
        perf = {}
    if not isinstance(deep, dict):
        deep = {}
    protocol = tuple(protocol_ids or ())
    protocol_mismatch = False
    if protocol:
        score_source, resolved, unresolved, empty, missing = _protocol_block(perf, deep, protocol)
        base_resolved, _, _, base_missing = partition(
            base.resolved_ids,
            base.unresolved_ids,
            base.empty_patch_ids,
            protocol,
        )
        if missing or base_missing:
            protocol_mismatch = True
            score = None
            base_score = base.score
        else:
            score = len(resolved) / len(protocol)
            base_score = len(base_resolved) / len(protocol)
        submitted = int((score_source or {}).get("total_submitted_instances") or 0)
    else:
        score_source = deep if deep.get("accuracy_score") is not None else perf
        score = score_source.get("accuracy_score") if score_source else None
        if score is not None:
            score = float(score)
        base_score = base.score
        resolved = tuple(score_source.get("total_resolved_ids") or perf.get("total_resolved_ids") or ())
        unresolved = tuple(score_source.get("total_unresolved_ids") or perf.get("total_unresolved_ids") or ())
        empty = tuple(score_source.get("total_emptypatch_ids") or perf.get("total_emptypatch_ids") or ())
        submitted = int(perf.get("total_submitted_instances") or 0) + int(deep.get("total_submitted_instances") or 0)
    patch_nonempty = bool(metadata.get("model_patch_notempty"))
    has_score = score is not None
    compiled = patch_nonempty and has_score and (len(resolved) + len(unresolved) > 0)

    lost = [item for item in base.resolved_ids if item in set(unresolved) or item in set(empty)]
    if protocol:
        lost = [item for item in lost if item in protocol]
    regression = (not protocol_mismatch) and (
        bool(lost) or (compiled and score is not None and score < base_score - 1e-12)
    )
    entry = metadata.get("entry")
    if entry and (entry in resolved or entry in unresolved or entry in empty):
        resolved_target: bool | None = entry in resolved
    else:
        resolved_target = None

    drop = (base_score - score) if (compiled and score is not None) else 0.0
    structural = bool(metadata.get("evopilot_structural")) or (
        compiled and not protocol_mismatch and drop >= catastrophic_drop
    )
    if protocol_mismatch and not metadata.get("evopilot_failure_cluster"):
        cluster = "incomparable_protocol"
    else:
        cluster = failure_cluster_for(
            compiled=compiled,
            structural=structural and not metadata.get("evopilot_failure_cluster"),
            lost=lost,
            resolved_target=resolved_target,
            entry=entry if isinstance(entry, str) else None,
            score=score,
            base_score=base_score,
            override=metadata.get("evopilot_failure_cluster"),
        )
    local_error = (not structural) and (not protocol_mismatch) and (
        not compiled
        or regression
        or resolved_target is False
        or (score is not None and score <= base_score + 1e-12)
    )
    files: tuple[str, ...] = ()
    if patch_path and os.path.exists(patch_path):
        try:
            with open(patch_path, "r", encoding="utf-8", errors="replace") as handle:
                files = parse_diff_paths(handle.read())
        except OSError:
            files = ()
    if not change_intent:
        statement = (metadata.get("problem_statement") or "").strip()
        change_intent = statement.splitlines()[0][:180] if statement else ""
        if not change_intent and entry:
            change_intent = f"self-edit targeting {entry}"

    n_evaluations = max(0, submitted - int(already_counted_evaluations))
    variance = extract_variance(metadata)
    if variance is None:
        variance = variance_from_result_files(results_dir, str(metadata.get("run_id") or ""))
    notes = ""
    if metadata.get("error"):
        notes = str(metadata.get("error"))[:500]
    return VerificationEvidence(
        candidate_id=str(metadata.get("run_id") or "unknown"),
        base_agent_id=base.agent_id,
        base_score=base_score,
        candidate_score=score,
        compiled=compiled,
        regression=regression,
        structural_error=structural,
        local_error=local_error,
        resolved_target=resolved_target,
        failure_cluster=cluster,
        change_intent=change_intent,
        targeted_failure_mode=str(entry or ""),
        files_or_mechanisms=files,
        observed_regression=",".join(lost) if lost else None,
        resolved_ids=tuple(resolved),
        unresolved_ids=tuple(unresolved),
        empty_patch_ids=tuple(empty),
        n_evaluations=n_evaluations,
        tokens=extract_tokens(metadata),
        sample_variance=variance,
        debug_round=debug_round,
        escalations_used=escalations_used,
        lineage_id=lineage_id,
        notes=notes,
        score_protocol=protocol,
        protocol_mismatch=protocol_mismatch,
    )


def _protocol_block(
    perf: dict,
    deep: dict,
    protocol: tuple[str, ...],
) -> tuple[dict, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Pick the report whose id lists cover ``protocol``, if one does.

    ``overall_performance`` is preferred so a small-subset report is not
    replaced by a later medium report that uses a different denominator.
    """

    best: tuple[dict, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None
    for block in (perf, deep):
        if not block:
            continue
        resolved, unresolved, empty, missing = partition(
            block.get("total_resolved_ids") or (),
            block.get("total_unresolved_ids") or (),
            block.get("total_emptypatch_ids") or (),
            protocol,
        )
        choice = (block, resolved, unresolved, empty, missing)
        if not missing:
            return choice
        if best is None or len(missing) < len(best[-1]):
            best = choice
    if best is None:
        return {}, (), (), (), protocol
    return best


def with_escalation_fallback(evidence: VerificationEvidence) -> VerificationEvidence:
    """Use when a broader re-evaluation cannot be run.

    The point estimate is kept and escalation is marked spent so the next
    decision cannot ask for the same extra sample again.
    """

    return replace(
        evidence,
        escalations_used=evidence.escalations_used + 1,
        sample_variance=0.0,
        n_evaluations=0,
        tokens=None,
    )
