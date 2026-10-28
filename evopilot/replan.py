"""Replan hooks.

After rollback or stagnation the next improve receives falsifiable
constraints, and a debug receives the open candidate's change intent plus
the verification evidence. These strings are appended to the existing
self-improve problem statement; they do not replace DGM's diagnose prompt
unless the caller passes an override.
"""

from __future__ import annotations

from evopilot.memory import EvolutionMemory
from evopilot.types import ReplanContext, VerificationEvidence


def build_replan_context(
    *,
    active_id: str,
    memory: EvolutionMemory,
    broaden: bool,
    reason: str,
    extra_constraints: tuple[str, ...] = (),
) -> ReplanContext:
    soft = list(memory.active_constraints())
    for item in extra_constraints:
        if item and item not in soft:
            soft.append(item)
    return ReplanContext(
        parent_id=active_id,
        excluded_intents=tuple(memory.recent_failed_intents()),
        soft_constraints=tuple(soft),
        broaden_scope=broaden,
        increase_verification=broaden,
        reason=reason,
    )


def format_improve_suffix(context: ReplanContext) -> str:
    if not context.soft_constraints and not context.excluded_intents and not context.broaden_scope:
        return ""
    lines = [
        "EvoPilot recovery constraints (soft and falsifiable; retire them if a later accepted edit contradicts them):"
    ]
    for constraint in context.soft_constraints:
        lines.append(f"- {constraint}")
    if context.excluded_intents:
        lines.append("Do not repeat these failed change intents:")
        for intent in context.excluded_intents:
            lines.append(f"- {intent}")
    if context.broaden_scope:
        lines.append(
            "Recent evolution looks stagnant. Propose a different direction, or split a change that mixed several effects."
        )
    if context.increase_verification:
        lines.append("The next candidate should be checked on a broader task sample before it is trusted.")
    return "\n".join(lines)


def format_debug_statement(evidence: VerificationEvidence, context: ReplanContext) -> str:
    files = ", ".join(evidence.files_or_mechanisms) if evidence.files_or_mechanisms else "(not recorded)"
    constraints = "\n".join(f"- {item}" for item in context.soft_constraints) or "- (none)"
    return (
        "Repair the current candidate in place. Do not discard it and do not start an unrelated rewrite.\n\n"
        f"Change intent: {evidence.change_intent or '(unspecified)'}\n"
        f"Targeted failure: {evidence.targeted_failure_mode or '(unspecified)'}\n"
        f"Failure cluster: {evidence.failure_cluster}\n"
        f"Files or mechanisms: {files}\n"
        f"Candidate score: {evidence.candidate_score}\n"
        f"Active score: {evidence.base_score}\n"
        f"Compiled: {evidence.compiled}\n"
        f"Regression: {evidence.regression}\n"
        f"Resolved target: {evidence.resolved_target}\n"
        f"Observed regression: {evidence.observed_regression or '(none)'}\n"
        f"Debug round already used: {evidence.debug_round}\n\n"
        "Verification says the direction is still plausible, but the implementation is locally wrong. "
        "Fix that failure. Keep the change small enough that a later regression can be attributed to it.\n\n"
        f"Soft constraints:\n{constraints}\n"
    )
