"""Verification-conditioned decision between accept, escalate, debug, and rollback.

The order is deliberate:

1. A structural or catastrophic regression is not repaired in place.
2. A candidate that did not compile can still take a local debug while the
   debug budget remains.
3. High sample variance on an ambiguous or regressing score escalates to a
   larger evaluation before the controller commits.
4. A compiled candidate that beats the active agent without a regression is
   accepted.
5. Otherwise a plausible candidate is debugged while ``r < R`` and evidence
   is still moving.
6. Exhausted or stalled repair rolls back.
"""

from __future__ import annotations

from dataclasses import dataclass

from evopilot.types import Decision, VerificationEvidence


@dataclass(frozen=True)
class VerifyPolicy:
    debug_budget: int = 2
    accept_margin: float = 0.0
    catastrophic_drop: float = 0.2
    variance_threshold: float = 0.05
    max_escalations: int = 1

    def decide(self, evidence: VerificationEvidence) -> tuple[Decision, str]:
        if evidence.structural_error or self._catastrophic(evidence):
            return Decision.ROLLBACK, "structural_or_catastrophic_regression"
        if evidence.protocol_mismatch:
            return Decision.ROLLBACK, "incomparable_evaluation_protocol"
        if not evidence.compiled:
            if self._can_debug(evidence):
                return Decision.DEBUG, "uncompiled_candidate_within_budget"
            if evidence.debug_stalled:
                return Decision.ROLLBACK, "uncompiled_and_debug_stalled"
            return Decision.ROLLBACK, "uncompiled_and_debug_budget_exhausted"
        if self._should_escalate(evidence):
            return Decision.ESCALATE, "sample_variance_exceeds_threshold"
        if self._should_accept(evidence):
            return Decision.ACCEPT, "score_improves_without_regression"
        if self._can_debug(evidence) and self._salvageable(evidence):
            return Decision.DEBUG, "direction_plausible_local_repair"
        if evidence.debug_stalled:
            return Decision.ROLLBACK, "debug_did_not_improve_evidence"
        if evidence.debug_round >= self.debug_budget:
            return Decision.ROLLBACK, "debug_budget_exhausted"
        return Decision.ROLLBACK, "no_accept_and_not_salvageable"

    def _can_debug(self, evidence: VerificationEvidence) -> bool:
        return (not evidence.debug_stalled) and evidence.debug_round < self.debug_budget

    def _catastrophic(self, evidence: VerificationEvidence) -> bool:
        if not evidence.compiled or evidence.candidate_score is None:
            return False
        return (evidence.base_score - evidence.candidate_score) >= self.catastrophic_drop

    def _should_accept(self, evidence: VerificationEvidence) -> bool:
        if evidence.regression or evidence.candidate_score is None:
            return False
        return evidence.candidate_score > evidence.base_score + self.accept_margin + 1e-12

    def _should_escalate(self, evidence: VerificationEvidence) -> bool:
        if evidence.sample_variance is None:
            return False
        if evidence.escalations_used >= self.max_escalations:
            return False
        if evidence.sample_variance <= self.variance_threshold:
            return False
        if evidence.candidate_score is None:
            return False
        gap = abs(evidence.candidate_score - evidence.base_score)
        ambiguous = gap <= max(self.accept_margin, self.variance_threshold)
        return ambiguous or evidence.regression

    def _salvageable(self, evidence: VerificationEvidence) -> bool:
        if evidence.candidate_score is None:
            return bool(evidence.local_error or not evidence.compiled)
        drop = evidence.base_score - evidence.candidate_score
        if drop >= self.catastrophic_drop:
            return False
        return True
