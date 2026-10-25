"""Control-oriented evolution memory.

Entries are soft constraints, not permanent rules. A later accepted edit
with the same intent is counterevidence: confidence drops, and the
constraint is retired after repeated counterevidence or when confidence
falls below the floor.
"""

from __future__ import annotations

import re
from dataclasses import asdict

from evopilot.types import MemoryEntry

_SPACE = re.compile(r"\s+")
CONFIDENCE_FLOOR = 0.25
INITIAL_ROLLBACK_CONFIDENCE = 0.6
COUNTEREVIDENCE_STEP = 0.3
RETIRE_AFTER = 2
REINFORCE_STEP = 0.1
CONFIDENCE_CAP = 0.95
UNTRIED_INTENT_LIMIT = 3


def normalize_intent(intent: str) -> str:
    return _SPACE.sub(" ", (intent or "").strip().lower())


class EvolutionMemory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []
        self._open: MemoryEntry | None = None

    @property
    def open_entry(self) -> MemoryEntry | None:
        return self._open

    def begin(
        self,
        *,
        entry_id: str,
        step: int,
        checkpoint_id: str,
        change_intent: str,
        targeted_failure_mode: str,
    ) -> MemoryEntry:
        if self._open is not None and self._open.outcome == "open":
            if change_intent:
                self._open.change_intent = change_intent
            if targeted_failure_mode:
                self._open.targeted_failure_mode = targeted_failure_mode
            return self._open
        entry = MemoryEntry(
            entry_id=entry_id,
            step=step,
            checkpoint_id=checkpoint_id,
            change_intent=change_intent,
            targeted_failure_mode=targeted_failure_mode,
        )
        self.entries.append(entry)
        self._open = entry
        return entry

    def record_verification(self, payload: dict) -> None:
        entry = self._require_open()
        entry.verification_evidence.append(payload)
        candidate_id = payload.get("candidate_id")
        if candidate_id and candidate_id not in entry.candidate_ids:
            entry.candidate_ids.append(candidate_id)
        cluster = payload.get("failure_cluster") or ""
        if cluster:
            entry.failure_cluster = cluster
        observed = payload.get("observed_regression")
        if observed:
            entry.observed_regression = observed
        files = payload.get("files_or_mechanisms") or []
        for name in files:
            if name not in entry.files_or_mechanisms:
                entry.files_or_mechanisms.append(name)
        if payload.get("change_intent"):
            entry.change_intent = payload["change_intent"]
        if payload.get("targeted_failure_mode"):
            entry.targeted_failure_mode = payload["targeted_failure_mode"]

    def record_debug(self, payload: dict) -> None:
        self._require_open().debug_attempts.append(payload)

    def close(
        self,
        *,
        outcome: str,
        reason: str,
        failure_cluster: str,
        rollback_target: str | None = None,
        rollback_reason: str | None = None,
    ) -> MemoryEntry:
        entry = self._require_open()
        entry.outcome = outcome
        entry.failure_attribution = reason
        entry.failure_cluster = failure_cluster or entry.failure_cluster
        entry.rollback_target = rollback_target
        entry.rollback_reason = rollback_reason
        if outcome == "rolled_back":
            entry.confidence = INITIAL_ROLLBACK_CONFIDENCE
            entry.future_constraint = (
                f"Avoid repeating intent '{entry.change_intent}' from checkpoint "
                f"{entry.checkpoint_id} when failure cluster is {entry.failure_cluster}."
            )
            self._reinforce(entry.checkpoint_id, entry.failure_cluster, skip=entry)
        else:
            entry.confidence = 1.0
            entry.future_constraint = None
        self._open = None
        return entry

    def attempt_count(self, checkpoint_id: str, failure_cluster: str) -> int:
        return sum(
            1
            for entry in self.entries
            if entry.outcome == "rolled_back"
            and not entry.retired
            and entry.checkpoint_id == checkpoint_id
            and entry.failure_cluster == failure_cluster
        )

    def has_untried_alternative(self, checkpoint_id: str, failure_cluster: str) -> bool:
        intents = {
            normalize_intent(entry.change_intent)
            for entry in self.entries
            if entry.checkpoint_id == checkpoint_id
            and entry.failure_cluster == failure_cluster
            and normalize_intent(entry.change_intent)
        }
        return len(intents) < UNTRIED_INTENT_LIMIT

    def precedes_failure(self, checkpoint_step: int, failure_cluster: str, current_step: int) -> bool:
        steps = [
            entry.step
            for entry in self.entries
            if entry.outcome == "rolled_back" and entry.failure_cluster == failure_cluster
        ]
        onset = min(steps) if steps else current_step
        return checkpoint_step < onset

    def observe_accepted_intent(self, intent: str) -> None:
        key = normalize_intent(intent)
        if not key:
            return
        for entry in self.entries:
            if entry.outcome != "rolled_back" or entry.retired or not entry.future_constraint:
                continue
            if normalize_intent(entry.change_intent) != key:
                continue
            entry.counterevidence += 1
            entry.confidence = max(0.0, entry.confidence - COUNTEREVIDENCE_STEP)
            if entry.counterevidence >= RETIRE_AFTER or entry.confidence < CONFIDENCE_FLOOR:
                entry.retired = True

    def active_constraints(self) -> list[str]:
        texts = []
        for entry in self.entries:
            if entry.retired or entry.outcome != "rolled_back" or not entry.future_constraint:
                continue
            if entry.confidence < CONFIDENCE_FLOOR:
                continue
            texts.append(entry.future_constraint)
        return texts

    def recent_failed_intents(self, limit: int = 5) -> list[str]:
        found: list[str] = []
        for entry in reversed(self.entries):
            if entry.outcome != "rolled_back" or entry.retired:
                continue
            intent = entry.change_intent.strip()
            if intent and intent not in found:
                found.append(intent)
            if len(found) >= limit:
                break
        return found

    def blocked_targets(self, min_confidence: float = 0.5) -> set[str]:
        blocked = set()
        for entry in self.entries:
            if entry.outcome != "rolled_back" or entry.retired:
                continue
            if entry.confidence < min_confidence:
                continue
            if entry.targeted_failure_mode:
                blocked.add(entry.targeted_failure_mode)
        return blocked

    def to_list(self) -> list[dict]:
        return [asdict(entry) for entry in self.entries]

    def load_list(self, rows: list[dict]) -> None:
        self.entries = [MemoryEntry(**row) for row in rows]
        self._open = next((entry for entry in reversed(self.entries) if entry.outcome == "open"), None)

    def _reinforce(self, checkpoint_id: str, failure_cluster: str, skip: MemoryEntry) -> None:
        for entry in self.entries:
            if entry is skip or entry.retired or entry.outcome != "rolled_back":
                continue
            if entry.checkpoint_id != checkpoint_id or entry.failure_cluster != failure_cluster:
                continue
            entry.confidence = min(CONFIDENCE_CAP, entry.confidence + REINFORCE_STEP)

    def _require_open(self) -> MemoryEntry:
        if self._open is None or self._open.outcome != "open":
            raise RuntimeError("no open evolution-memory attempt")
        return self._open
