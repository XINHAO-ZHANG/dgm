"""Single-active recovery controller.

The outer loop calls ``begin_attempt`` for an improve, then ``step`` once
per verification. Debug and escalate leave the attempt open. Accept commits
the candidate into the checkpoint bank. Rollback selects a bank member and
the next improve is replanned from that agent. Checkmate ends the lineage
and switches to an earlier checkpoint, the same hop stagnation escalation
uses.

``decider`` replaces ``VerifyPolicy`` for that one choice. The policy
remains the fallback when no decider is installed, and for host failures
passed with ``force_policy``. A decider's action is not rewritten from
the candidate score.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from evopilot.bank import CheckpointBank, RecoverabilityWeights
from evopilot.memory import EvolutionMemory
from evopilot.metrics import ComputeLedger
from evopilot.policy import VerifyPolicy
from evopilot.replan import build_replan_context
from evopilot.scoring import align_verification
from evopilot.stagnation import detect_stagnation
from evopilot.types import (
    AgentSnapshot,
    AttemptOutcome,
    ControlDecision,
    ControlResult,
    Decision,
    ReplanContext,
    StagnationSignal,
    VerificationEvidence,
    agent_from_dict,
    no_stagnation,
)


class RecoveryController:
    def __init__(
        self,
        policy: VerifyPolicy | None = None,
        bank_size: int = 4,
        *,
        stagnation_window: int = 4,
        stagnation_min_steps: int = 3,
        weights: RecoverabilityWeights | None = None,
        dead_end_threshold: int = 2,
    ) -> None:
        self.policy = policy or VerifyPolicy()
        self.bank = CheckpointBank(
            capacity=bank_size,
            weights=weights,
            dead_end_threshold=dead_end_threshold,
        )
        self.memory = EvolutionMemory()
        self.ledger = ComputeLedger()
        self.stagnation_window = stagnation_window
        self.stagnation_min_steps = stagnation_min_steps
        self.generation = 0
        self.lineage_id = ""
        self.active: AgentSnapshot | None = None
        self.best_score = 0.0
        self.best_agent_id = ""
        self.attempts: list[AttemptOutcome] = []
        self.last_stagnation: StagnationSignal = no_stagnation()
        self.flushed_events = 0
        self._best_before = 0.0
        self._debug_count = 0
        self._last_evidence: VerificationEvidence | None = None
        self._last_rollback_mode: str | None = None
        self._stagnation_latched = False
        self._accepted_last = False
        self._replan_reason = "steady"
        self.decider = None
        self.decision_source = "policy"
        self.workspace_dir: str | None = None
        self.output_dir: str | None = None
        self._agent_constraints: list[str] = []
        self._broaden_once = False

    def bootstrap(self, initial: AgentSnapshot) -> None:
        initial = replace(initial, root=True, step=0)
        self.active = initial
        self.best_score = initial.score
        self.best_agent_id = initial.agent_id
        self.bank = CheckpointBank(
            capacity=self.bank.capacity,
            weights=self.bank.weights,
            dead_end_threshold=self.bank.dead_end_threshold,
        )
        self.bank.admit(initial, active_id=initial.agent_id)
        self.ledger.record(
            event="bootstrap",
            step=0,
            active_score=initial.score,
            best_score=initial.score,
            candidate_score=None,
            decision=None,
            agent_id=initial.agent_id,
            evaluations=0,
            tokens=None,
            details={"bank": [initial.agent_id]},
        )

    def begin_attempt(self, change_intent: str, targeted_failure_mode: str = "") -> str:
        if self.active is None:
            raise RuntimeError("bootstrap the controller before an improve attempt")
        # The previous checkmate asked for one broader improve. The caller
        # reads replan_context before begin_attempt, so this consumes it.
        self._broaden_once = False
        open_entry = self.memory.open_entry
        if open_entry is not None and open_entry.outcome == "open":
            self.memory.begin(
                entry_id=open_entry.entry_id,
                step=self.generation,
                checkpoint_id=self.active.agent_id,
                change_intent=change_intent,
                targeted_failure_mode=targeted_failure_mode,
            )
            return self.lineage_id
        self.generation += 1
        self.lineage_id = f"lin-{self.generation}"
        self._best_before = self.best_score
        self._debug_count = 0
        self._last_evidence = None
        self.memory.begin(
            entry_id=f"mem-{self.generation}-{self.active.agent_id}",
            step=self.generation,
            checkpoint_id=self.active.agent_id,
            change_intent=change_intent,
            targeted_failure_mode=targeted_failure_mode,
        )
        return self.lineage_id

    def step(self, evidence: VerificationEvidence, *, force_policy: bool = False) -> ControlResult:
        if self.active is None:
            raise RuntimeError("bootstrap the controller before verification")
        origin = self.active
        evidence = self._prepare_evidence(evidence, origin)
        if self.memory.open_entry is None:
            self.begin_attempt(evidence.change_intent, evidence.targeted_failure_mode)

        constraints: tuple[str, ...] = ()
        use_agent = self.decider is not None and not force_policy
        if use_agent:
            chosen = self.decider(evidence)
            if not isinstance(chosen, ControlDecision):
                raise RuntimeError("control decider must return ControlDecision")
            decision = chosen.action
            reason = chosen.reason
            constraints = chosen.next_constraints
            if decision not in {Decision.ACCEPT, Decision.DEBUG, Decision.ROLLBACK, Decision.CHECKMATE}:
                raise RuntimeError(f"control decider returned {decision}; score rules will not replace it")
        else:
            decision, reason = self.policy.decide(evidence)
        decision_source = "llm" if use_agent else "policy"
        self.memory.record_verification(self._verification_payload(evidence, decision, reason))

        rollback_target = None
        rollback_mode = None
        bank_reason = None
        if decision == Decision.ACCEPT:
            self._accept(evidence, origin)
            self._last_rollback_mode = None
            self._stagnation_latched = False
            self._accepted_last = True
            self._replan_reason = "accepted"
        elif decision == Decision.DEBUG:
            self._debug_count += 1
            self.memory.record_debug(
                {
                    "round": evidence.debug_round,
                    "candidate_id": evidence.candidate_id,
                    "score": evidence.candidate_score,
                    "failure_cluster": evidence.failure_cluster,
                    "outcome": "debug",
                }
            )
            self._replan_reason = reason
        elif decision == Decision.ROLLBACK:
            rollback_target, rollback_mode, bank_reason = self._commit_bank_move(
                origin,
                evidence,
                reason,
                force_deeper=False,
                memory_outcome="rolled_back",
                attempt_decision="rollback",
            )
        elif decision == Decision.CHECKMATE:
            rollback_target, rollback_mode, bank_reason = self._commit_bank_move(
                origin,
                evidence,
                reason,
                force_deeper=True,
                memory_outcome="checkmate",
                attempt_decision="checkmate",
            )
            self._stagnation_latched = True
            self._broaden_once = True
            self._replan_reason = reason
        elif decision == Decision.ESCALATE:
            self._replan_reason = reason
        else:
            raise RuntimeError(f"unexpected decision {decision}")

        self._remember_constraints(constraints, replace=decision == Decision.ACCEPT)

        if decision == Decision.ACCEPT:
            closed = self.memory.close(
                outcome="accepted",
                reason=reason,
                failure_cluster=evidence.failure_cluster,
            )
            self.memory.observe_accepted_intent(closed.change_intent)
            self._record_outcome(closed.change_intent, evidence.failure_cluster, "accept")

        self._last_evidence = evidence
        self.last_stagnation = detect_stagnation(
            self.attempts,
            window=self.stagnation_window,
            min_steps=self.stagnation_min_steps,
        )
        next_debug = evidence.debug_round + 1 if decision == Decision.DEBUG else evidence.debug_round
        if decision in {Decision.ACCEPT, Decision.ROLLBACK, Decision.CHECKMATE}:
            next_debug = 0

        assert self.active is not None
        details = {
            "reason": reason,
            "failure_cluster": evidence.failure_cluster,
            "lineage_id": evidence.lineage_id,
            "bank": [agent.agent_id for agent in self.bank.ordered()],
            "compiled": evidence.compiled,
            "regression": evidence.regression,
            "resolved_target": evidence.resolved_target,
            "debug_round": evidence.debug_round,
            "debug_stalled": evidence.debug_stalled,
            "decision_source": decision_source,
            "next_constraints": list(constraints),
        }
        if rollback_target is not None:
            details["rollback_target"] = rollback_target.agent_id
            details["rollback_mode"] = rollback_mode
            details["rollback_reason"] = bank_reason
        self.ledger.record(
            event=decision.value,
            step=self.generation,
            active_score=self.active.score,
            best_score=self.best_score,
            candidate_score=evidence.candidate_score,
            decision=decision.value,
            agent_id=self.active.agent_id,
            evaluations=evidence.n_evaluations,
            tokens=evidence.tokens,
            details=details,
        )
        return self._result(
            decision,
            reason,
            rollback_target,
            rollback_mode,
            next_debug,
            constraints,
        )

    def maybe_escape_stagnation(self) -> ControlResult | None:
        """On a newly detected plateau, roll back once to an earlier checkpoint.

        A deeper rollback on the previous attempt already changed targets, so
        this only latches replan mode (broader scope, excluded intents, more
        verification) and does not take a second hop in the same window.
        """

        if self.active is None:
            raise RuntimeError("bootstrap the controller before stagnation checks")
        signal = detect_stagnation(
            self.attempts,
            window=self.stagnation_window,
            min_steps=self.stagnation_min_steps,
        )
        self.last_stagnation = signal
        # The check runs at the start of the next improve. A candidate that
        # was just accepted must be allowed to evolve before another
        # stagnation rollback undoes it.
        if self._accepted_last:
            self._accepted_last = False
            return None
        if not signal.stagnant:
            self._stagnation_latched = False
            self._replan_reason = "steady"
            return None
        if self._stagnation_latched:
            return None

        self._stagnation_latched = True
        moved = False
        target_id = self.active.agent_id
        mode = None
        cluster = signal.dominant_cluster or "stagnation"
        if self._last_rollback_mode != "deeper":
            choice = self.bank.select(
                active=self.active,
                cluster=cluster,
                memory=self.memory,
                current_step=self.generation,
                force_deeper=True,
            )
            mode = choice.mode
            if choice.target.agent_id != self.active.agent_id:
                tagged = self.active.with_failure_cluster(cluster)
                self.bank.admit(tagged, active_id=choice.target.agent_id)
                self.active = choice.target
                moved = True
                self._last_rollback_mode = "deeper"
            target_id = choice.target.agent_id
            self._replan_reason = "stagnation_deeper_rollback" if moved else "stagnation_replan"
        else:
            self._replan_reason = "stagnation_after_deeper_rollback"

        self.ledger.record(
            event="replan",
            step=self.generation,
            active_score=self.active.score,
            best_score=self.best_score,
            candidate_score=None,
            decision=None,
            agent_id=self.active.agent_id,
            evaluations=0,
            tokens=None,
            details={
                "reason": self._replan_reason,
                "moved": moved,
                "rollback_target": target_id,
                "rollback_mode": mode,
                "stagnation_reasons": list(signal.reasons),
                "dominant_cluster": signal.dominant_cluster,
                "bank": [agent.agent_id for agent in self.bank.ordered()],
            },
        )
        return self._result(Decision.ROLLBACK, self._replan_reason, self.active, mode, 0)

    def close_bounded_debug(self) -> None:
        """End an open lineage when Debug was requested and the bound is spent.

        The recorded decision stays debug. The bank is left as it is and the
        candidate score is not consulted.
        """

        entry = self.memory.open_entry
        if entry is None or entry.outcome != "open":
            return
        cluster = self._last_evidence.failure_cluster if self._last_evidence else ""
        intent = entry.change_intent
        self.memory.close(
            outcome="debug_bounded",
            reason="debug_execution_bound",
            failure_cluster=cluster,
        )
        self._record_outcome(intent, cluster, "debug")

    def replan_context(self) -> ReplanContext:
        if self.active is None:
            raise RuntimeError("bootstrap the controller before replanning")
        broaden = self._broaden_once or self._stagnation_latched or self.last_stagnation.stagnant
        return build_replan_context(
            active_id=self.active.agent_id,
            memory=self.memory,
            broaden=broaden,
            reason=self._replan_reason,
            extra_constraints=tuple(self._agent_constraints),
        )

    def suggest_entry(self) -> str | None:
        if self.active is None:
            return None
        blocked = self.memory.blocked_targets()
        pool = list(self.active.unresolved_ids) or list(self.active.resolved_ids)
        for entry in pool:
            if entry not in blocked:
                return entry
        return pool[0] if pool else None

    def summary(self) -> dict:
        auec = self.ledger.auec()
        salvage_rate, salvaged, failed = self._salvage_stats()
        return {
            "generation": self.generation,
            "active": asdict(self.active) if self.active else None,
            "best_score": self.best_score,
            "best_agent_id": self.best_agent_id,
            "bank_ids": [agent.agent_id for agent in self.bank.ordered()],
            "cumulative_evaluations": self.ledger.cumulative_evaluations,
            "cumulative_tokens": self.ledger.cumulative_tokens,
            "decisions": [event.get("decision") for event in self.ledger.events if event.get("decision")],
            "salvage_rate": salvage_rate,
            "salvage_accepted_after_debug": salvaged,
            "salvage_failed_first_verify": failed,
            "decision_source": "llm" if self.decider is not None else "policy",
            **auec,
        }

    def _salvage_stats(self) -> tuple[float | None, int, int]:
        """Fraction of first-verify failures later accepted after debug.

        An immediate accept did not fail verification. A rollback, or an
        accept that used at least one debug round, counts as a first-verify
        failure. The rate is null when that denominator is zero.
        """

        failed = 0
        salvaged = 0
        for attempt in self.attempts:
            if attempt.decision == "accept" and attempt.debug_count <= 0:
                continue
            failed += 1
            if attempt.decision == "accept" and attempt.debug_count > 0:
                salvaged += 1
        if failed == 0:
            return None, 0, 0
        return salvaged / failed, salvaged, failed

    def to_dict(self) -> dict:
        return {
            "policy": {
                "debug_budget": self.policy.debug_budget,
                "accept_margin": self.policy.accept_margin,
                "catastrophic_drop": self.policy.catastrophic_drop,
                "variance_threshold": self.policy.variance_threshold,
                "max_escalations": self.policy.max_escalations,
            },
            "bank_size": self.bank.capacity,
            "dead_end_threshold": self.bank.dead_end_threshold,
            "weights": asdict(self.bank.weights),
            "stagnation_window": self.stagnation_window,
            "stagnation_min_steps": self.stagnation_min_steps,
            "generation": self.generation,
            "lineage_id": self.lineage_id,
            "active": asdict(self.active) if self.active else None,
            "best_score": self.best_score,
            "best_agent_id": self.best_agent_id,
            "bank": self.bank.to_list(),
            "memory": self.memory.to_list(),
            "attempts": [asdict(item) for item in self.attempts],
            "last_rollback_mode": self._last_rollback_mode,
            "stagnation_latched": self._stagnation_latched,
            "accepted_last": self._accepted_last,
            "replan_reason": self._replan_reason,
            "debug_count": self._debug_count,
            "best_before": self._best_before,
            "decision_source": self.decision_source,
            "agent_constraints": list(self._agent_constraints),
            "broaden_once": self._broaden_once,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryController":
        policy = VerifyPolicy(**data["policy"])
        weights = RecoverabilityWeights(**data["weights"])
        controller = cls(
            policy=policy,
            bank_size=int(data["bank_size"]),
            stagnation_window=int(data["stagnation_window"]),
            stagnation_min_steps=int(data["stagnation_min_steps"]),
            weights=weights,
            dead_end_threshold=int(data["dead_end_threshold"]),
        )
        controller.generation = int(data["generation"])
        controller.lineage_id = data.get("lineage_id") or ""
        controller.best_score = float(data["best_score"])
        controller.best_agent_id = data.get("best_agent_id") or ""
        controller.bank.load_list(data.get("bank") or [])
        controller.memory.load_list(data.get("memory") or [])
        controller.attempts = [AttemptOutcome(**row) for row in data.get("attempts") or []]
        controller.active = agent_from_dict(data["active"]) if data.get("active") else None
        controller._last_rollback_mode = data.get("last_rollback_mode")
        controller._stagnation_latched = bool(data.get("stagnation_latched", False))
        controller._accepted_last = bool(data.get("accepted_last", False))
        controller._replan_reason = data.get("replan_reason") or "steady"
        controller.decision_source = data.get("decision_source") or "policy"
        controller._agent_constraints = [str(item) for item in data.get("agent_constraints") or [] if str(item).strip()]
        controller._broaden_once = bool(data.get("broaden_once", False))
        controller._debug_count = int(data.get("debug_count") or 0)
        controller._best_before = float(data.get("best_before") or controller.best_score)
        controller.last_stagnation = detect_stagnation(
            controller.attempts,
            window=controller.stagnation_window,
            min_steps=controller.stagnation_min_steps,
        )
        return controller

    def _prepare_evidence(self, evidence: VerificationEvidence, origin: AgentSnapshot) -> VerificationEvidence:
        evidence = replace(
            evidence,
            base_agent_id=origin.agent_id,
            lineage_id=evidence.lineage_id or self.lineage_id,
            n_evaluations=max(0, int(evidence.n_evaluations)),
        )
        if evidence.score_protocol:
            evidence = align_verification(evidence, origin)
        else:
            evidence = replace(evidence, base_score=origin.score)
        previous = self._last_evidence
        if (
            previous is not None
            and previous.lineage_id == evidence.lineage_id
            and evidence.debug_round > previous.debug_round
        ):
            improved = False
            if evidence.compiled and not previous.compiled:
                improved = True
            if (
                evidence.candidate_score is not None
                and previous.candidate_score is not None
                and evidence.candidate_score > previous.candidate_score + 1e-9
            ):
                improved = True
            if evidence.resolved_target is True and previous.resolved_target is not True:
                improved = True
            same_cluster = evidence.failure_cluster == previous.failure_cluster
            stalled = (not improved) and same_cluster
            evidence = replace(evidence, debug_stalled=evidence.debug_stalled or stalled)
        return evidence

    def _commit_bank_move(
        self,
        origin: AgentSnapshot,
        evidence: VerificationEvidence,
        reason: str,
        *,
        force_deeper: bool,
        memory_outcome: str,
        attempt_decision: str,
    ):
        choice = self.bank.select(
            active=origin,
            cluster=evidence.failure_cluster,
            memory=self.memory,
            current_step=self.generation,
            force_deeper=force_deeper,
        )
        self._apply_rollback(origin, choice, evidence.failure_cluster)
        closed = self.memory.close(
            outcome=memory_outcome,
            reason=reason,
            failure_cluster=evidence.failure_cluster,
            rollback_target=choice.target.agent_id,
            rollback_reason=choice.reason,
        )
        self._record_outcome(closed.change_intent, evidence.failure_cluster, attempt_decision)
        self._last_rollback_mode = choice.mode
        self._accepted_last = False
        if not force_deeper:
            self._replan_reason = choice.reason
        return choice.target, choice.mode, choice.reason

    def _remember_constraints(self, constraints: tuple[str, ...], *, replace: bool) -> None:
        if replace:
            self._agent_constraints = []
        for item in constraints:
            text = " ".join(str(item).split())
            if text and text not in self._agent_constraints:
                self._agent_constraints.append(text)
        if len(self._agent_constraints) > 8:
            self._agent_constraints = self._agent_constraints[-8:]

    def _accept(self, evidence: VerificationEvidence, origin: AgentSnapshot) -> None:
        if evidence.candidate_score is None:
            raise RuntimeError("cannot accept a candidate without a score")
        snapshot = AgentSnapshot(
            agent_id=evidence.candidate_id,
            parent_id=origin.agent_id,
            score=float(evidence.candidate_score),
            step=self.generation,
            resolved_ids=evidence.resolved_ids,
            unresolved_ids=evidence.unresolved_ids,
            empty_patch_ids=evidence.empty_patch_ids,
            failure_clusters=(),
            root=False,
        )
        self.bank.admit(snapshot, active_id=snapshot.agent_id)
        self.active = snapshot
        if snapshot.score > self.best_score + 1e-12:
            self.best_score = snapshot.score
            self.best_agent_id = snapshot.agent_id

    def _apply_rollback(self, origin: AgentSnapshot, choice, cluster: str) -> None:
        if choice.target.agent_id != origin.agent_id:
            tagged = origin.with_failure_cluster(cluster)
            self.bank.admit(tagged, active_id=choice.target.agent_id)
        self.active = choice.target

    def _record_outcome(self, intent: str, cluster: str, decision: str) -> None:
        assert self.active is not None
        self.attempts.append(
            AttemptOutcome(
                step=self.generation,
                intent=intent,
                failure_cluster=cluster,
                decision=decision,
                debug_count=self._debug_count,
                best_before=self._best_before,
                best_after=self.best_score,
                active_after=self.active.score,
            )
        )

    def _verification_payload(self, evidence: VerificationEvidence, decision: Decision, reason: str) -> dict:
        return {
            "candidate_id": evidence.candidate_id,
            "score": evidence.candidate_score,
            "base_score": evidence.base_score,
            "compiled": evidence.compiled,
            "regression": evidence.regression,
            "structural_error": evidence.structural_error,
            "failure_cluster": evidence.failure_cluster,
            "n_evaluations": evidence.n_evaluations,
            "tokens": evidence.tokens,
            "sample_variance": evidence.sample_variance,
            "debug_round": evidence.debug_round,
            "debug_stalled": evidence.debug_stalled,
            "decision": decision.value,
            "reason": reason,
            "observed_regression": evidence.observed_regression,
            "files_or_mechanisms": list(evidence.files_or_mechanisms),
            "change_intent": evidence.change_intent,
            "targeted_failure_mode": evidence.targeted_failure_mode,
        }

    def _result(
        self,
        decision: Decision,
        reason: str,
        rollback_target: AgentSnapshot | None,
        rollback_mode: str | None,
        debug_round: int,
        next_constraints: tuple[str, ...] = (),
    ) -> ControlResult:
        assert self.active is not None
        return ControlResult(
            decision=decision,
            reason=reason,
            active=self.active,
            rollback_target=rollback_target,
            rollback_mode=rollback_mode,
            debug_round=debug_round,
            lineage_id=self.lineage_id,
            replan=self.replan_context(),
            stagnation=self.last_stagnation,
            next_constraints=next_constraints,
        )
