"""Deterministic dry-run trace for the recovery controller.

The scores below are fixtures. They are not Polyglot measurements and they
are not copied from ``initial_polyglot/metadata.json``. The trace is here
so a reviewer can exercise accept, bounded debug, local rollback, deeper
rollback, and the AUEC ledger without API keys or Docker.
"""

from __future__ import annotations

from evopilot.controller import RecoveryController
from evopilot.policy import VerifyPolicy
from evopilot.reporting import flush_run
from evopilot.types import AgentSnapshot, Decision, VerificationEvidence

DRY_RUN_DISCLAIMER = (
    "Synthetic controller trace for smoke testing. Not a Polyglot or SWE-bench result, "
    "and not a reported paper metric."
)

INITIAL = AgentSnapshot(
    agent_id="initial",
    parent_id=None,
    score=0.40,
    step=0,
    resolved_ids=("python__beer-song",),
    unresolved_ids=("go__poker", "cpp__clock"),
    root=True,
)

# Each attempt is one improve. Extra rounds are debug verifications of the
# same candidate lineage. Absolute scores are candidate scores.
SCENARIO = [
    {
        "intent": "cache repeated tool results",
        "entry": "go__poker",
        "rounds": [
            {"candidate_id": "c1", "score": 0.50, "cluster": "ok", "resolved_target": True, "evals": 10},
        ],
    },
    {
        "intent": "tighten the edit parser",
        "entry": "cpp__clock",
        "rounds": [
            {
                "candidate_id": "c2a",
                "score": 0.47,
                "cluster": "regression:go",
                "regression": True,
                "local_error": True,
                "resolved_target": False,
                "evals": 10,
            },
            {
                "candidate_id": "c2",
                "score": 0.55,
                "cluster": "ok",
                "resolved_target": True,
                "evals": 10,
            },
        ],
    },
    {
        "intent": "rewrite the planner",
        "entry": "go__poker",
        "rounds": [
            {
                "candidate_id": "c3",
                "score": 0.10,
                "cluster": "structural_planner",
                "structural": True,
                "compiled": True,
                "evals": 1,
            },
        ],
    },
    {
        "intent": "retry the go regression",
        "entry": "go__poker",
        "rounds": [
            {
                "candidate_id": "c4",
                "score": 0.20,
                "cluster": "regression:go",
                "structural": True,
                "compiled": True,
                "evals": 1,
            },
        ],
    },
    {
        "intent": "retry the go regression",
        "entry": "go__poker",
        "rounds": [
            {
                "candidate_id": "c5",
                "score": 0.20,
                "cluster": "regression:go",
                "structural": True,
                "compiled": True,
                "evals": 1,
            },
        ],
    },
    {
        "intent": "retry the go regression",
        "entry": "go__poker",
        "rounds": [
            {
                "candidate_id": "c6",
                "score": 0.20,
                "cluster": "regression:go",
                "structural": True,
                "compiled": True,
                "evals": 1,
            },
        ],
    },
    {
        "intent": "split context handling from tool parsing",
        "entry": "cpp__clock",
        "rounds": [
            {"candidate_id": "c7", "score": 0.52, "cluster": "ok", "resolved_target": True, "evals": 10},
        ],
    },
]


def make_evidence(controller: RecoveryController, attempt: dict, rnd: dict, debug_round: int) -> VerificationEvidence:
    assert controller.active is not None
    return VerificationEvidence(
        candidate_id=rnd["candidate_id"],
        base_agent_id=controller.active.agent_id,
        base_score=controller.active.score,
        candidate_score=rnd.get("score"),
        compiled=rnd.get("compiled", True),
        regression=rnd.get("regression", False),
        structural_error=rnd.get("structural", False),
        local_error=rnd.get("local_error", False),
        resolved_target=rnd.get("resolved_target"),
        failure_cluster=rnd["cluster"],
        change_intent=attempt["intent"],
        targeted_failure_mode=attempt.get("entry", ""),
        n_evaluations=rnd.get("evals", 0),
        tokens=rnd.get("tokens"),
        debug_round=debug_round,
        lineage_id=controller.lineage_id,
        resolved_ids=tuple(rnd.get("resolved_ids", controller.active.resolved_ids)),
        unresolved_ids=tuple(rnd.get("unresolved_ids", controller.active.unresolved_ids)),
    )


def run_smoke(
    output_dir: str,
    *,
    max_attempts: int | None = None,
    bank_size: int = 4,
    debug_budget: int = 2,
) -> dict:
    controller = RecoveryController(
        policy=VerifyPolicy(debug_budget=debug_budget),
        bank_size=bank_size,
    )
    controller.bootstrap(INITIAL)
    attempts = SCENARIO if max_attempts is None else SCENARIO[: max(0, max_attempts)]
    for attempt in attempts:
        controller.maybe_escape_stagnation()
        controller.begin_attempt(attempt["intent"], attempt.get("entry", ""))
        result = None
        for rnd in attempt["rounds"]:
            if result is not None and result.decision != Decision.DEBUG:
                break
            debug_round = 0 if result is None else result.debug_round
            evidence = make_evidence(controller, attempt, rnd, debug_round)
            result = controller.step(evidence)
    flush_run(output_dir, controller, mode="dry_run", disclaimer=DRY_RUN_DISCLAIMER)
    return controller.summary() | {"mode": "dry_run", "disclaimer": DRY_RUN_DISCLAIMER}
