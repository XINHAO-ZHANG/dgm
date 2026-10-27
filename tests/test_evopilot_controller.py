"""Controller decisions for accept, debug, escalate, and rollback.

These tests do not call a model or the Polyglot harness.
"""

import ast
import json
import statistics
from dataclasses import replace
from pathlib import Path

import pytest

import EvoPilot_outer
from EvoPilot_outer import (
    build_evidence,
    call_self_improve,
    live_start_message,
    main,
    snapshot_from_initial_metadata,
)
from evopilot.scoring import evaluation_protocol, verify_schedule
from evopilot.bank import CheckpointBank, RecoverabilityWeights, recoverability_score
from evopilot.controller import RecoveryController
from evopilot.evidence import evidence_from_dgm_metadata, parse_diff_paths
from evopilot.memory import EvolutionMemory
from evopilot.metrics import ComputeLedger
from evopilot.policy import VerifyPolicy
from evopilot.replan import format_debug_statement, format_improve_suffix
from evopilot.reporting import flush_run, load_run
from evopilot.types import AgentSnapshot, Decision, MemoryEntry, VerificationEvidence


def boot(**controller_kwargs) -> RecoveryController:
    controller = RecoveryController(**controller_kwargs)
    controller.bootstrap(
        AgentSnapshot(
            agent_id="initial",
            parent_id=None,
            score=0.40,
            step=0,
            resolved_ids=("python__beer-song",),
            unresolved_ids=("go__poker", "cpp__clock"),
            root=True,
        )
    )
    return controller


def evidence(controller: RecoveryController, **overrides) -> VerificationEvidence:
    base = dict(
        candidate_id="cand",
        base_agent_id=controller.active.agent_id,
        base_score=controller.active.score,
        candidate_score=0.50,
        compiled=True,
        regression=False,
        structural_error=False,
        local_error=False,
        resolved_target=True,
        failure_cluster="ok",
        change_intent="cache tool results",
        targeted_failure_mode="go__poker",
        n_evaluations=10,
        debug_round=0,
        lineage_id=controller.lineage_id,
        resolved_ids=controller.active.resolved_ids,
        unresolved_ids=controller.active.unresolved_ids,
    )
    base.update(overrides)
    return VerificationEvidence(**base)


def test_policy_accept_debug_escalate_and_rollback():
    policy = VerifyPolicy(debug_budget=2, catastrophic_drop=0.2, variance_threshold=0.05)
    base = VerificationEvidence(
        candidate_id="c",
        base_agent_id="initial",
        base_score=0.50,
        candidate_score=0.60,
        compiled=True,
        regression=False,
        structural_error=False,
        local_error=False,
        resolved_target=True,
        failure_cluster="ok",
        change_intent="improve",
        targeted_failure_mode="task",
    )
    assert policy.decide(base) == (Decision.ACCEPT, "score_improves_without_regression")
    assert policy.decide(replace(base, candidate_score=0.70, sample_variance=0.2))[0] == Decision.ACCEPT
    assert policy.decide(replace(base, candidate_score=0.52, sample_variance=0.2)) == (
        Decision.ESCALATE,
        "sample_variance_exceeds_threshold",
    )
    assert policy.decide(replace(base, candidate_score=0.45, regression=True, local_error=True))[0] == Decision.DEBUG
    assert policy.decide(replace(base, compiled=False, candidate_score=None, local_error=True))[0] == Decision.DEBUG
    rolled = policy.decide(replace(base, compiled=False, candidate_score=None, debug_round=2))
    assert rolled == (Decision.ROLLBACK, "uncompiled_and_debug_budget_exhausted")
    assert policy.decide(replace(base, candidate_score=0.10))[0] == Decision.ROLLBACK
    assert policy.decide(replace(base, structural_error=True, candidate_score=0.49))[1] == (
        "structural_or_catastrophic_regression"
    )
    stalled = policy.decide(replace(base, candidate_score=0.45, regression=True, debug_round=1, debug_stalled=True))
    assert stalled == (Decision.ROLLBACK, "debug_did_not_improve_evidence")


def test_controller_accepts_a_better_compiled_candidate():
    controller = boot()
    controller.begin_attempt("cache tool results", "go__poker")
    result = controller.step(evidence(controller, candidate_id="c1", candidate_score=0.50))
    assert result.decision == Decision.ACCEPT
    assert result.active.agent_id == "c1"
    assert controller.best_agent_id == "c1"
    assert [agent.agent_id for agent in controller.bank.ordered()] == ["initial", "c1"]


def test_debug_then_rollback_when_evidence_does_not_improve():
    controller = boot()
    controller.begin_attempt("tighten parser", "cpp__clock")
    first = controller.step(
        evidence(
            controller,
            candidate_id="bad",
            candidate_score=0.38,
            regression=True,
            local_error=True,
            resolved_target=False,
            failure_cluster="regression:cpp",
        )
    )
    assert first.decision == Decision.DEBUG
    assert controller.active.agent_id == "initial"
    second = controller.step(
        evidence(
            controller,
            candidate_id="bad2",
            candidate_score=0.38,
            regression=True,
            local_error=True,
            resolved_target=False,
            failure_cluster="regression:cpp",
            debug_round=first.debug_round,
        )
    )
    assert second.decision == Decision.ROLLBACK
    assert second.reason == "debug_did_not_improve_evidence"
    assert second.rollback_mode == "local"
    assert second.active.agent_id == "initial"


def test_escalate_before_committing_an_ambiguous_score():
    controller = boot()
    controller.begin_attempt("ambiguous", "go__poker")
    result = controller.step(evidence(controller, candidate_score=0.42, sample_variance=0.2, resolved_target=None))
    assert result.decision == Decision.ESCALATE
    follow = controller.step(
        evidence(
            controller,
            candidate_score=0.55,
            sample_variance=0.2,
            escalations_used=1,
            n_evaluations=5,
        )
    )
    assert follow.decision == Decision.ACCEPT
    assert controller.ledger.cumulative_evaluations == 15


def test_recent_accept_is_not_immediately_abandoned():
    controller = boot()
    controller.begin_attempt("cache tool results", "go__poker")
    controller.step(evidence(controller, candidate_id="c1", candidate_score=0.50))
    assert controller.maybe_escape_stagnation() is None
    assert controller.active.agent_id == "c1"


def test_deeper_rollback_after_repeated_failures_from_the_same_checkpoint():
    controller = boot()
    controller.begin_attempt("first gain", "go__poker")
    controller.step(evidence(controller, candidate_id="c1", candidate_score=0.50))
    controller.begin_attempt("second gain", "cpp__clock")
    controller.step(evidence(controller, candidate_id="c2", candidate_score=0.55))
    modes = []
    for index in range(3):
        controller.begin_attempt("retry the go regression", "go__poker")
        result = controller.step(
            evidence(
                controller,
                candidate_id=f"bad{index}",
                candidate_score=0.20,
                structural_error=True,
                failure_cluster="regression:go",
                n_evaluations=1,
            )
        )
        modes.append(result.rollback_mode)
    assert modes == ["local", "local", "deeper"]
    assert controller.active.agent_id == "c1"
    assert controller.best_agent_id == "c2"


def test_stagnation_escape_moves_once_then_latches():
    controller = boot(dead_end_threshold=100, stagnation_window=3, stagnation_min_steps=3)
    controller.begin_attempt("gain", "go__poker")
    controller.step(evidence(controller, candidate_id="child", candidate_score=0.60))
    for index in range(3):
        controller.begin_attempt("same idea", "go__poker")
        result = controller.step(
            evidence(
                controller,
                candidate_id=f"bad{index}",
                candidate_score=0.20,
                structural_error=True,
                failure_cluster="regression:go",
                n_evaluations=1,
            )
        )
        assert result.rollback_mode == "local"
        assert controller.active.agent_id == "child"
    escape = controller.maybe_escape_stagnation()
    assert escape is not None
    assert controller.active.agent_id == "initial"
    assert controller.maybe_escape_stagnation() is None
    assert controller.active.agent_id == "initial"
    replans = [event for event in controller.ledger.events if event["event"] == "replan"]
    assert len(replans) == 1
    assert replans[0]["details"]["moved"] is True


def test_bank_keeps_root_best_and_latest_within_capacity():
    bank = CheckpointBank(capacity=4)
    bank.admit(AgentSnapshot(agent_id="root", parent_id=None, score=0.10, step=0, root=True))
    bank.admit(AgentSnapshot(agent_id="a1", parent_id="root", score=0.50, step=1))
    bank.admit(AgentSnapshot(agent_id="a2", parent_id="root", score=0.40, step=2))
    bank.admit(AgentSnapshot(agent_id="a3", parent_id="root", score=0.30, step=3))
    bank.admit(AgentSnapshot(agent_id="a4", parent_id="root", score=0.20, step=4))
    assert [agent.agent_id for agent in bank.ordered()] == ["root", "a1", "a2", "a4"]

    tiny = CheckpointBank(capacity=1)
    tiny.admit(AgentSnapshot(agent_id="root", parent_id=None, score=0.10, step=0, root=True))
    tiny.admit(AgentSnapshot(agent_id="latest", parent_id="root", score=0.20, step=1))
    assert [agent.agent_id for agent in tiny.ordered()] == ["latest"]


def test_recoverability_score_matches_the_weighted_terms():
    weights = RecoverabilityWeights()
    memory = EvolutionMemory()
    for step in (4, 5):
        memory.entries.append(
            MemoryEntry(
                entry_id=f"m{step}",
                step=step,
                checkpoint_id="c2",
                change_intent="retry the go regression",
                targeted_failure_mode="go__poker",
                failure_cluster="regression:go",
                outcome="rolled_back",
            )
        )
    active = AgentSnapshot(agent_id="c2", parent_id="c1", score=0.55, step=2)
    earlier = AgentSnapshot(agent_id="c1", parent_id="initial", score=0.50, step=1)
    score = recoverability_score(
        earlier,
        active=active,
        cluster="regression:go",
        memory=memory,
        weights=weights,
        dead_end_threshold=2,
        current_step=6,
    )
    expected = (
        weights.capability * 0.50
        + weights.precedes_failure
        + weights.per_step_distance * 1
        + weights.untried_alternative
    )
    assert score == pytest.approx(expected)

    bank = CheckpointBank(capacity=4, weights=weights)
    bank.admit(AgentSnapshot(agent_id="initial", parent_id=None, score=0.40, step=0, root=True))
    bank.admit(earlier)
    bank.admit(active)
    choice = bank.select(active=active, cluster="regression:go", memory=memory, current_step=6)
    assert choice.mode == "deeper"
    assert choice.target.agent_id == "c1"
    fresh = EvolutionMemory()
    local = bank.select(active=active, cluster="regression:go", memory=fresh, current_step=6)
    assert local.mode == "local"
    assert local.target.agent_id == "c2"


def test_soft_constraint_retires_after_counterevidence_and_entry_is_skipped():
    controller = boot()
    controller.begin_attempt("Rewrite Parser", "go__poker")
    controller.step(
        evidence(
            controller,
            candidate_id="bad",
            structural_error=True,
            failure_cluster="missed:go__poker",
            change_intent="Rewrite Parser",
        )
    )
    constraints = controller.memory.active_constraints()
    assert any("Rewrite Parser" in text for text in constraints)
    assert controller.suggest_entry() == "cpp__clock"
    suffix = format_improve_suffix(controller.replan_context())
    assert "Rewrite Parser" in suffix

    controller.begin_attempt("rewrite parser", "cpp__clock")
    controller.step(evidence(controller, candidate_id="c1", candidate_score=0.50, change_intent="rewrite parser"))
    assert controller.memory.entries[0].counterevidence == 1
    assert controller.memory.entries[0].retired is False

    controller.begin_attempt("  Rewrite   parser ", "cpp__clock")
    controller.step(
        evidence(controller, candidate_id="c2", candidate_score=0.60, change_intent="  Rewrite   parser ")
    )
    assert controller.memory.entries[0].retired is True
    assert controller.memory.active_constraints() == []


def test_debug_statement_keeps_the_change_intent():
    controller = boot()
    item = evidence(controller, change_intent="tighten the edit parser", failure_cluster="regression:go")
    text = format_debug_statement(item, controller.replan_context())
    assert "tighten the edit parser" in text
    assert "Repair the current candidate in place" in text


def test_auec_uses_the_step_curve_after_each_event():
    ledger = ComputeLedger()
    common = dict(candidate_score=None, decision=None, agent_id="initial", tokens=None, details={})
    ledger.record(event="bootstrap", step=0, active_score=0.10, best_score=0.10, evaluations=0, **common)
    ledger.record(event="verify", step=1, active_score=0.10, best_score=0.10, evaluations=10, **common)
    ledger.record(event="accept", step=2, active_score=0.40, best_score=0.50, evaluations=10, **common)
    report = ledger.auec()
    assert report["auec_best"] == pytest.approx(2.0)
    assert report["auec_active"] == pytest.approx(2.0)
    assert report["budget_evaluations"] == 20
    assert report["normalized_auec_best"] == pytest.approx(0.1)


def test_evidence_adapter_reads_dgm_metadata_without_inventing_tokens(tmp_path):
    diff = tmp_path / "model_patch.diff"
    diff.write_text("diff --git a/coding_agent.py b/coding_agent.py\n", encoding="utf-8")
    base = AgentSnapshot(
        agent_id="initial",
        parent_id=None,
        score=0.40,
        step=0,
        resolved_ids=("python__beer-song", "go__poker"),
        unresolved_ids=("cpp__clock",),
    )
    metadata = {
        "run_id": "child",
        "entry": "cpp__clock",
        "model_patch_notempty": True,
        "problem_statement": "Fix the clock parser\nmore",
        "token_usage": {"input_tokens": 3, "output_tokens": 4},
        "per_eval_scores": [0.2, 0.4],
        "overall_performance": {
            "accuracy_score": 0.45,
            "total_submitted_instances": 10,
            "total_resolved_ids": ["python__beer-song"],
            "total_unresolved_ids": ["go__poker", "cpp__clock"],
            "total_emptypatch_ids": [],
        },
    }
    built = evidence_from_dgm_metadata(metadata, base=base, patch_path=str(diff), already_counted_evaluations=4)
    assert built.compiled is True
    assert built.regression is True
    assert built.failure_cluster == "regression:go"
    assert built.resolved_target is False
    assert built.tokens == 7
    assert built.n_evaluations == 6
    assert built.sample_variance == pytest.approx(statistics.pstdev([0.2, 0.4]))
    assert built.files_or_mechanisms == ("coding_agent.py",)
    assert built.change_intent == "Fix the clock parser"
    assert parse_diff_paths("diff --git a/foo b/foo\ndiff --git a/foo b/foo\n") == ("foo",)

    empty = evidence_from_dgm_metadata({"run_id": "none", "model_patch_notempty": False}, base=base)
    assert empty.compiled is False
    assert empty.tokens is None
    assert empty.failure_cluster == "compile_or_empty_patch"

    (tmp_path / "child_0.000.json").write_text(
        json.dumps({"submitted_instances": 10, "resolved_instances": 2}),
        encoding="utf-8",
    )
    (tmp_path / "child_1.000.json").write_text(
        json.dumps({"submitted_instances": 10, "resolved_instances": 8}),
        encoding="utf-8",
    )
    spread = evidence_from_dgm_metadata(
        {"run_id": "child", "model_patch_notempty": False},
        base=base,
        results_dir=str(tmp_path),
    )
    assert spread.sample_variance == pytest.approx(statistics.pstdev([0.2, 0.8]))

    crashed = evidence_from_dgm_metadata(
        {
            "run_id": "failed-1",
            "model_patch_notempty": False,
            "evopilot_structural": True,
            "evopilot_failure_cluster": "infrastructure_error",
        },
        base=base,
    )
    assert crashed.structural_error is True
    assert crashed.failure_cluster == "infrastructure_error"


def test_dry_run_writes_auec_curve_and_can_resume(tmp_path):
    output = tmp_path / "run"
    code = main(["--dry_run", "--output_dir", str(output), "--max_generation", "80"])
    assert code == 0
    events = [
        json.loads(line)
        for line in (output / "evopilot_trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [event["decision"] for event in events if event.get("decision")]
    assert decisions == [
        "accept",
        "debug",
        "accept",
        "rollback",
        "rollback",
        "rollback",
        "rollback",
        "accept",
    ]
    rollbacks = [event for event in events if event.get("decision") == "rollback"]
    assert [event["details"]["rollback_mode"] for event in rollbacks] == ["local", "local", "local", "deeper"]
    assert rollbacks[-1]["details"]["rollback_target"] == "c1"
    assert any(event["event"] == "replan" and event["details"]["moved"] is False for event in events)

    summary = json.loads((output / "evopilot_summary.json").read_text(encoding="utf-8"))
    assert summary["disclaimer"].startswith("Synthetic")
    assert summary["cumulative_tokens"] is None
    assert summary["auec_best"] == pytest.approx(21.7)
    assert summary["auec_active"] == pytest.approx(21.2)
    assert summary["auec_active"] < summary["auec_best"]
    assert summary["best_agent_id"] == "c2"
    assert summary["active"]["agent_id"] == "c7"
    assert set(summary["bank_ids"]) == {"initial", "c1", "c2", "c7"}
    assert summary["salvage_accepted_after_debug"] == 1
    assert summary["salvage_failed_first_verify"] == 5
    assert summary["salvage_rate"] == pytest.approx(0.2)

    resumed = load_run(str(output))
    assert resumed.active.agent_id == "c7"
    assert resumed.best_score == pytest.approx(0.55)
    before = len((output / "evopilot_trajectory.jsonl").read_text(encoding="utf-8").splitlines())
    flush_run(str(output), resumed, mode="dry_run", disclaimer=summary["disclaimer"])
    after = len((output / "evopilot_trajectory.jsonl").read_text(encoding="utf-8").splitlines())
    assert after == before


def test_second_dry_run_replaces_the_trajectory_file(tmp_path):
    assert main(["--dry_run", "--output_dir", str(tmp_path), "--max_generation", "1"]) == 0
    assert main(["--dry_run", "--output_dir", str(tmp_path), "--max_generation", "80"]) == 0
    events = [
        json.loads(line)
        for line in (tmp_path / "evopilot_trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bootstraps = [event for event in events if event["event"] == "bootstrap"]
    assert len(bootstraps) == 1
    assert events[-1]["cumulative_evaluations"] == 44
    assert events[0]["cumulative_evaluations"] == 0


def test_dry_run_respects_max_generation(tmp_path):
    output = tmp_path / "short"
    assert main(["--dry_run", "--output_dir", str(output), "--max_generation", "1"]) == 0
    summary = json.loads((output / "evopilot_summary.json").read_text(encoding="utf-8"))
    assert summary["decisions"] == ["accept"]
    assert summary["salvage_rate"] is None
    assert summary["salvage_failed_first_verify"] == 0


def test_infrastructure_failure_does_not_require_docker(monkeypatch):
    real_import = __import__

    def broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "self_improve_step":
            raise ModuleNotFoundError("docker")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", broken_import)
    metadata = call_self_improve(parent_commit="initial", entry="go__poker")
    assert metadata["evopilot_structural"] is True
    assert metadata["evopilot_failure_cluster"] == "infrastructure_error"
    assert metadata["run_id"].startswith("failed-")


def test_dgm_outer_baseline_is_untouched():
    text = Path("DGM_outer.py").read_text(encoding="utf-8")
    assert "evopilot" not in text
    assert "def main(" in text
    assert "def choose_selfimproves(" in text
    source = Path("self_improve_step.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "self_improve")
    names = [arg.arg for arg in function.args.args]
    assert names[-3:] == ["problem_statement_override", "problem_statement_suffix", "evopilot_role"]
    assert len(function.args.defaults) == len(names)


def test_outer_module_imports_without_the_live_stack():
    assert callable(EvoPilot_outer.main)


def test_live_start_message_selects_apptainer_without_a_docker_daemon():
    text = live_start_message("apptainer", "polyglot")
    assert "runtime apptainer" in text
    assert "DGM_CONTAINER_RUNTIME=apptainer" in text
    assert "no Docker daemon is available" in text
    assert "Docker are required" not in text
    assert "Docker is required" not in text
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("## EvoPilot recovery controller", 1)[1].split("## File Structure", 1)[0]
    assert "DGM_CONTAINER_RUNTIME=apptainer" in section
    assert "working Docker daemon" not in section


def _protocol_metadata(protocol, resolved, accuracy_score):
    resolved_set = set(resolved)
    return {
        "run_id": "child",
        "entry": protocol[0],
        "model_patch_notempty": True,
        "overall_performance": {
            "accuracy_score": accuracy_score,
            "total_submitted_instances": len(protocol),
            "total_resolved_ids": list(resolved),
            "total_unresolved_ids": [item for item in protocol if item not in resolved_set],
            "total_emptypatch_ids": [],
        },
    }


def test_full_suite_baseline_is_not_compared_with_a_subset_score():
    """2/10 = 0.20 beats a full-suite 0.142 and still is not an improvement.

    The cached initial agent resolves 2 of the 10 small Polyglot tasks, so
    its comparable accuracy is 0.20. A candidate with the same two resolves
    must not be accepted, including when its report claims a higher
    ``accuracy_score`` than the id lists support. One extra resolve on that
    same set must be accepted even if the report's raw fraction is 0.142.
    """

    small = json.loads(Path("polyglot/subsets/small.json").read_text(encoding="utf-8"))
    medium = json.loads(Path("polyglot/subsets/medium.json").read_text(encoding="utf-8"))
    shallow = evaluation_protocol(small, medium, shallow=True)
    wide = evaluation_protocol(small, medium, shallow=False)
    assert len(shallow) == 10
    assert set(wide) == set(small) | set(medium)
    assert len(wide) == len(set(wide))

    initial = snapshot_from_initial_metadata("initial_polyglot/metadata.json", shallow)
    full = json.loads(Path("initial_polyglot/metadata.json").read_text(encoding="utf-8"))
    full_score = full["overall_performance"]["accuracy_score"]
    assert full_score == pytest.approx(0.142)
    assert initial.score == pytest.approx(0.2)
    assert initial.score > full_score
    covered = set(initial.resolved_ids) | set(initial.unresolved_ids) | set(initial.empty_patch_ids)
    assert covered == set(shallow)

    wide_initial = snapshot_from_initial_metadata("initial_polyglot/metadata.json", wide)
    assert wide_initial.score == pytest.approx(9 / 59)
    assert wide_initial.score != pytest.approx(full_score)

    # The unaligned comparison is the bug: 0.20 > 0.142 accepts.
    naive = VerifyPolicy().decide(
        VerificationEvidence(
            candidate_id="naive",
            base_agent_id="initial",
            base_score=full_score,
            candidate_score=0.20,
            compiled=True,
            regression=False,
            structural_error=False,
            local_error=False,
            resolved_target=None,
            failure_cluster="ok",
            change_intent="subset edit",
            targeted_failure_mode=shallow[0],
        )
    )
    assert naive[0] == Decision.ACCEPT

    controller = RecoveryController()
    controller.bootstrap(initial)
    same = _protocol_metadata(shallow, initial.resolved_ids, accuracy_score=0.99)
    built = build_evidence(
        same,
        controller,
        debug_round=0,
        escalations_used=0,
        already_counted=0,
        patch_path=None,
        protocol_ids=shallow,
    )
    assert built.base_score == pytest.approx(0.2)
    assert built.candidate_score == pytest.approx(0.2)
    assert built.protocol_mismatch is False
    held = controller.step(built)
    assert held.decision != Decision.ACCEPT
    assert controller.active.score == pytest.approx(0.2)

    # The controller also ignores a lying candidate_score when the protocol is set.
    controller = RecoveryController()
    controller.bootstrap(
        AgentSnapshot(
            agent_id="initial",
            parent_id=None,
            score=full_score,
            step=0,
            resolved_ids=tuple(full["overall_performance"]["total_resolved_ids"]),
            unresolved_ids=tuple(full["overall_performance"]["total_unresolved_ids"]),
            empty_patch_ids=tuple(full["overall_performance"]["total_emptypatch_ids"]),
            root=True,
        )
    )
    lying = evidence(
        controller,
        candidate_score=0.20,
        base_score=full_score,
        score_protocol=shallow,
        resolved_ids=tuple(initial.resolved_ids),
        unresolved_ids=tuple(initial.unresolved_ids),
        empty_patch_ids=tuple(initial.empty_patch_ids),
        regression=False,
        failure_cluster="ok",
        resolved_target=None,
    )
    blocked = controller.step(lying)
    assert blocked.decision != Decision.ACCEPT
    assert controller.ledger.events[-1]["candidate_score"] == pytest.approx(0.2)
    assert controller.ledger.events[-1]["details"]["reason"] != "score_improves_without_regression"

    extras = list(initial.unresolved_ids) + list(initial.empty_patch_ids)
    assert extras
    gained = list(initial.resolved_ids) + [extras[0]]
    improved = _protocol_metadata(shallow, gained, accuracy_score=full_score)
    controller = RecoveryController()
    controller.bootstrap(initial)
    accepted = controller.step(
        build_evidence(
            improved,
            controller,
            debug_round=0,
            escalations_used=0,
            already_counted=0,
            patch_path=None,
            protocol_ids=shallow,
        )
    )
    assert accepted.decision == Decision.ACCEPT
    assert accepted.active.score == pytest.approx(0.3)
    assert accepted.active.score != pytest.approx(full_score)
    stored = set(accepted.active.resolved_ids) | set(accepted.active.unresolved_ids) | set(accepted.active.empty_patch_ids)
    assert stored == set(shallow)

    partial = _protocol_metadata(shallow, initial.resolved_ids, accuracy_score=0.90)
    mismatched = controller.step(
        build_evidence(
            partial,
            controller,
            debug_round=0,
            escalations_used=0,
            already_counted=0,
            patch_path=None,
            protocol_ids=wide,
        )
    )
    assert mismatched.decision == Decision.ROLLBACK
    assert mismatched.reason == "incomparable_evaluation_protocol"
    assert controller.active.score == pytest.approx(0.3)
    assert controller.best_score == pytest.approx(0.3)


def test_incomplete_baseline_cache_is_not_scored_with_the_full_suite_fraction(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "overall_performance": {
                    "accuracy_score": 0.9,
                    "total_resolved_ids": ["a"],
                    "total_unresolved_ids": [],
                    "total_emptypatch_ids": [],
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="full-suite"):
        snapshot_from_initial_metadata(str(path), ("a", "b"))

    small = json.loads(Path("swe_bench/subsets/small.json").read_text(encoding="utf-8"))
    medium = json.loads(Path("swe_bench/subsets/medium.json").read_text(encoding="utf-8"))
    shallow = snapshot_from_initial_metadata(
        "initial/metadata.json",
        evaluation_protocol(small, medium, shallow=True),
    )
    assert shallow.score == pytest.approx(0.3)
    with pytest.raises(RuntimeError, match="no cached result"):
        snapshot_from_initial_metadata(
            "initial/metadata.json",
            evaluation_protocol(small, medium, shallow=False),
        )


def test_default_verify_schedule_matches_dgm_polyglot_staging():
    """Live Polyglot starts on small.json and promotes to medium.json at 0.4.

    ``--shallow_eval`` keeps the screen and does not open the second stage.
    Accept stays on the screen, so a candidate that never promotes is still
    comparable. The medium stage is extra submitted instances.
    """

    small = json.loads(Path("polyglot/subsets/small.json").read_text(encoding="utf-8"))
    medium = json.loads(Path("polyglot/subsets/medium.json").read_text(encoding="utf-8"))
    assert len(small) == 10
    assert len(medium) == 50
    staged = verify_schedule(small, medium, shallow=False, threshold=0.4)
    assert staged.test_task_list == tuple(small)
    assert staged.test_more_threshold == 0.4
    assert staged.test_task_list_more == tuple(medium)
    assert staged.score_protocol == tuple(small)
    smoke = verify_schedule(small, medium, shallow=True, threshold=0.4)
    assert smoke.test_more_threshold is None
    assert smoke.test_task_list_more is None
    assert smoke.score_protocol == tuple(small)

    dgm = Path("DGM_outer.py").read_text(encoding="utf-8")
    assert "test_more_threshold = 0.4" in dgm
    assert "test_more_threshold=None if args.shallow_eval else test_more_threshold" in dgm
    assert "test_task_list_more=None if args.shallow_eval else swe_issues_med" in dgm
    harness = Path("self_improve_step.py").read_text(encoding="utf-8")
    assert ">= len(test_task_list) * test_more_threshold" in harness
    outer = Path("EvoPilot_outer.py").read_text(encoding="utf-8")
    assert "verify_schedule(" in outer
    assert "test_more_threshold=schedule.test_more_threshold" in outer
    assert "test_more_threshold=None" not in outer

    initial = snapshot_from_initial_metadata("initial_polyglot/metadata.json", staged.score_protocol)
    assert initial.score == pytest.approx(0.2)
    controller = RecoveryController()
    controller.bootstrap(initial)
    screen = list(staged.score_protocol)
    resolved = list(initial.resolved_ids)
    held = _protocol_metadata(screen, resolved, accuracy_score=0.99)
    held["overall_performance_deep"] = {
        "accuracy_score": 0.2,
        "total_submitted_instances": len(medium),
        "total_resolved_ids": resolved[:1],
        "total_unresolved_ids": [],
        "total_emptypatch_ids": [],
    }
    built = build_evidence(
        held,
        controller,
        debug_round=0,
        escalations_used=0,
        already_counted=0,
        patch_path=None,
        protocol_ids=staged.score_protocol,
    )
    assert built.protocol_mismatch is False
    assert built.candidate_score == pytest.approx(initial.score)
    assert built.n_evaluations == len(small) + len(medium)
    decision = controller.step(built)
    assert decision.decision != Decision.ACCEPT
    assert decision.reason != "incomparable_evaluation_protocol"
