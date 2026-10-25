"""Small trusted checkpoint bank and explicit recoverability scores.

Capacity K defaults to 4 (the design uses about 3–5). Eviction never drops
the protected set, in this order, up to K slots: the most recently accepted
checkpoint, the best-scoring checkpoint, the active agent, then the root.
Other slots keep higher scores.

Rollback target selection:

- Local rollback returns the active checkpoint when the failure is not
  already in that checkpoint and fewer than ``dead_end_threshold`` rollbacks
  on this failure cluster have started there.
- Otherwise the active checkpoint is excluded when another checkpoint
  exists, and the remainder is ranked by recoverability.

Recoverability(H_j) =
    w_capability * score(H_j)
    + w_precedes if H_j was accepted before this failure cluster was first rolled back
    + w_exhibits if this cluster was already attributed to H_j
    + w_retry * (rolled-back attempts that started at H_j on this cluster)
    + w_dead if that attempt count reaches the dead-end threshold
    + w_distance * (active.step - H_j.step)
    + w_untried if fewer than three distinct intents were tried from H_j on this cluster
"""

from __future__ import annotations

from dataclasses import dataclass

from evopilot.memory import EvolutionMemory
from evopilot.types import AgentSnapshot, RollbackChoice


@dataclass(frozen=True)
class RecoverabilityWeights:
    capability: float = 1.0
    precedes_failure: float = 0.25
    exhibits_failure: float = -0.35
    per_failed_attempt: float = -0.15
    dead_end: float = -0.50
    per_step_distance: float = -0.05
    untried_alternative: float = 0.20


def exhibits_failure(agent: AgentSnapshot, cluster: str) -> bool:
    if not cluster or cluster in {"ok", "none"}:
        return False
    return cluster in agent.failure_clusters


def recoverability_score(
    agent: AgentSnapshot,
    *,
    active: AgentSnapshot,
    cluster: str,
    memory: EvolutionMemory,
    weights: RecoverabilityWeights,
    dead_end_threshold: int,
    current_step: int,
) -> float:
    retries = memory.attempt_count(agent.agent_id, cluster)
    distance = max(0, active.step - agent.step)
    score = weights.capability * agent.score
    if memory.precedes_failure(agent.step, cluster, current_step):
        score += weights.precedes_failure
    if exhibits_failure(agent, cluster):
        score += weights.exhibits_failure
    score += weights.per_failed_attempt * retries
    if retries >= dead_end_threshold:
        score += weights.dead_end
    score += weights.per_step_distance * distance
    if memory.has_untried_alternative(agent.agent_id, cluster):
        score += weights.untried_alternative
    return score


class CheckpointBank:
    def __init__(
        self,
        capacity: int = 4,
        *,
        weights: RecoverabilityWeights | None = None,
        dead_end_threshold: int = 2,
    ) -> None:
        if capacity < 1:
            raise ValueError("checkpoint bank capacity must be >= 1")
        if dead_end_threshold < 1:
            raise ValueError("dead_end_threshold must be >= 1")
        self.capacity = capacity
        self.weights = weights or RecoverabilityWeights()
        self.dead_end_threshold = dead_end_threshold
        self._items: dict[str, AgentSnapshot] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._items)

    def ordered(self) -> list[AgentSnapshot]:
        return [self._items[agent_id] for agent_id in self._order]

    def get(self, agent_id: str) -> AgentSnapshot | None:
        return self._items.get(agent_id)

    def admit(self, agent: AgentSnapshot, *, active_id: str | None = None) -> None:
        if agent.agent_id in self._items:
            self._items[agent.agent_id] = agent
            return
        self._items[agent.agent_id] = agent
        self._order.append(agent.agent_id)
        self._evict(active_id or agent.agent_id)

    def select(
        self,
        *,
        active: AgentSnapshot,
        cluster: str,
        memory: EvolutionMemory,
        current_step: int,
        force_deeper: bool = False,
    ) -> RollbackChoice:
        if not self._items:
            return RollbackChoice(active, "bank_empty_keep_active", "local", ())

        if active.agent_id not in self._items:
            self.admit(active, active_id=active.agent_id)

        scores = {
            agent.agent_id: recoverability_score(
                agent,
                active=active,
                cluster=cluster,
                memory=memory,
                weights=self.weights,
                dead_end_threshold=self.dead_end_threshold,
                current_step=current_step,
            )
            for agent in self.ordered()
        }
        score_rows = tuple(sorted(scores.items()))
        retries = memory.attempt_count(active.agent_id, cluster)
        attributable = not exhibits_failure(active, cluster)
        if (
            not force_deeper
            and attributable
            and retries < self.dead_end_threshold
            and active.agent_id in self._items
        ):
            return RollbackChoice(
                self._items[active.agent_id],
                "failure_attributed_to_latest_update",
                "local",
                score_rows,
            )

        pool = [agent for agent in self.ordered() if agent.agent_id != active.agent_id]
        if not pool:
            pool = self.ordered()

        def sort_key(agent: AgentSnapshot) -> tuple[float, int, int]:
            distance = max(0, active.step - agent.step)
            return (scores[agent.agent_id], -distance, agent.step)

        target = max(pool, key=sort_key)
        return RollbackChoice(
            target,
            "highest_recoverability_among_earlier_checkpoints",
            "deeper",
            score_rows,
        )

    def to_list(self) -> list[dict]:
        from dataclasses import asdict

        return [asdict(agent) for agent in self.ordered()]

    def load_list(self, rows: list[dict]) -> None:
        from evopilot.types import agent_from_dict

        self._items = {}
        self._order = []
        for row in rows:
            agent = agent_from_dict(row)
            self._items[agent.agent_id] = agent
            self._order.append(agent.agent_id)

    def _evict(self, active_id: str) -> None:
        while len(self._items) > self.capacity:
            protected = self._protected(active_id)
            victims = [agent_id for agent_id in self._order if agent_id not in protected]
            if not victims:
                break
            victim = min(victims, key=lambda agent_id: (self._items[agent_id].score, self._items[agent_id].step))
            del self._items[victim]
            self._order = [agent_id for agent_id in self._order if agent_id != victim]

    def _protected(self, active_id: str | None) -> set[str]:
        if not self._order:
            return set()
        latest = self._order[-1]
        best = max(self._items, key=lambda agent_id: (self._items[agent_id].score, self._items[agent_id].step))
        roots = [agent_id for agent_id in self._order if self._items[agent_id].root]
        ranked: list[str] = []
        for agent_id in (latest, best, active_id, *roots):
            if agent_id and agent_id in self._items and agent_id not in ranked:
                ranked.append(agent_id)
        return set(ranked[: self.capacity])
