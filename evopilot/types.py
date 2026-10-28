"""State objects for the EvoPilot recovery controller.

S_t = (H_t, C_t, M_t, E_t): the active agent, a small trusted checkpoint
bank, control-oriented evolution memory, and the current verification
evidence. Actions are improve, verify, debug, accept, escalate, rollback,
checkmate, and replan. Checkmate ends the current lineage and switches
checkpoint the way stagnation escalation already does. Improve itself stays
in the existing self-improve step. The default live decider is the Control
Agent; ``VerifyPolicy`` remains the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    ACCEPT = "accept"
    ESCALATE = "escalate"
    DEBUG = "debug"
    ROLLBACK = "rollback"
    CHECKMATE = "checkmate"


@dataclass(frozen=True)
class ControlDecision:
    """Structured Control Agent output. The outer loop executes ``action``."""

    action: Decision
    reason: str
    next_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentSnapshot:
    """A trusted or candidate agent version the controller can name."""

    agent_id: str
    parent_id: str | None
    score: float
    step: int
    resolved_ids: tuple[str, ...] = ()
    unresolved_ids: tuple[str, ...] = ()
    empty_patch_ids: tuple[str, ...] = ()
    failure_clusters: tuple[str, ...] = ()
    root: bool = False

    def with_failure_cluster(self, cluster: str) -> "AgentSnapshot":
        if not cluster or cluster in {"ok", "none"} or cluster in self.failure_clusters:
            return self
        return AgentSnapshot(
            agent_id=self.agent_id,
            parent_id=self.parent_id,
            score=self.score,
            step=self.step,
            resolved_ids=self.resolved_ids,
            unresolved_ids=self.unresolved_ids,
            empty_patch_ids=self.empty_patch_ids,
            failure_clusters=self.failure_clusters + (cluster,),
            root=self.root,
        )


def agent_from_dict(data: dict) -> AgentSnapshot:
    return AgentSnapshot(
        agent_id=data["agent_id"],
        parent_id=data.get("parent_id"),
        score=float(data["score"]),
        step=int(data["step"]),
        resolved_ids=tuple(data.get("resolved_ids") or ()),
        unresolved_ids=tuple(data.get("unresolved_ids") or ()),
        empty_patch_ids=tuple(data.get("empty_patch_ids") or ()),
        failure_clusters=tuple(data.get("failure_clusters") or ()),
        root=bool(data.get("root", False)),
    )


@dataclass(frozen=True)
class VerificationEvidence:
    """E_t for one verify of a candidate, including a debug round."""

    candidate_id: str
    base_agent_id: str
    base_score: float
    candidate_score: float | None
    compiled: bool
    regression: bool
    structural_error: bool
    local_error: bool
    resolved_target: bool | None
    failure_cluster: str
    change_intent: str
    targeted_failure_mode: str
    files_or_mechanisms: tuple[str, ...] = ()
    observed_regression: str | None = None
    resolved_ids: tuple[str, ...] = ()
    unresolved_ids: tuple[str, ...] = ()
    empty_patch_ids: tuple[str, ...] = ()
    n_evaluations: int = 0
    tokens: int | None = None
    sample_variance: float | None = None
    debug_round: int = 0
    debug_stalled: bool = False
    escalations_used: int = 0
    lineage_id: str = ""
    notes: str = ""
    # Problems both scores must cover. Empty means the caller already
    # supplied comparable floats (the dry-run fixtures do this).
    score_protocol: tuple[str, ...] = ()
    protocol_mismatch: bool = False


@dataclass
class MemoryEntry:
    """One improve attempt, including its debug rounds and final outcome.

    Field names follow the control-memory schema: checkpoint, change intent,
    files or mechanisms, targeted failure, verification evidence, observed
    regression, debug attempts, failure attribution, outcome, rollback target
    and reason, future constraint, confidence, and counterevidence.
    """

    entry_id: str
    step: int
    checkpoint_id: str
    change_intent: str
    targeted_failure_mode: str
    failure_cluster: str = ""
    failure_attribution: str = ""
    observed_regression: str | None = None
    outcome: str = "open"
    rollback_target: str | None = None
    rollback_reason: str | None = None
    future_constraint: str | None = None
    confidence: float = 0.5
    counterevidence: int = 0
    retired: bool = False
    candidate_ids: list[str] = field(default_factory=list)
    files_or_mechanisms: list[str] = field(default_factory=list)
    verification_evidence: list[dict] = field(default_factory=list)
    debug_attempts: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class AttemptOutcome:
    step: int
    intent: str
    failure_cluster: str
    decision: str
    debug_count: int
    best_before: float
    best_after: float
    active_after: float


@dataclass(frozen=True)
class StagnationSignal:
    stagnant: bool
    reasons: tuple[str, ...]
    delta_q: float
    novelty: float
    rollback_rate: float
    dominant_cluster: str | None
    dominant_cluster_fraction: float
    rollback_loop: bool


def no_stagnation() -> StagnationSignal:
    return StagnationSignal(
        stagnant=False,
        reasons=(),
        delta_q=0.0,
        novelty=1.0,
        rollback_rate=0.0,
        dominant_cluster=None,
        dominant_cluster_fraction=0.0,
        rollback_loop=False,
    )


@dataclass(frozen=True)
class ReplanContext:
    parent_id: str
    excluded_intents: tuple[str, ...]
    soft_constraints: tuple[str, ...]
    broaden_scope: bool
    increase_verification: bool
    reason: str


@dataclass(frozen=True)
class RollbackChoice:
    target: AgentSnapshot
    reason: str
    mode: str
    scores: tuple[tuple[str, float], ...]


@dataclass
class ControlResult:
    decision: Decision
    reason: str
    active: AgentSnapshot
    rollback_target: AgentSnapshot | None
    rollback_mode: str | None
    debug_round: int
    lineage_id: str
    replan: ReplanContext
    stagnation: StagnationSignal
    next_constraints: tuple[str, ...] = ()
