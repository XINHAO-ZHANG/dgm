"""Control Agent decisions. The model is mocked; these tests do not call an API."""

import json

import pytest

import EvoPilot_outer
from EvoPilot_outer import install_controller_decider, parse_args
from evopilot.control_agent import (
    CONTROL_AGENT_SYSTEM_PROMPT,
    ControlAgent,
    ControlAgentParseError,
    bind_control_agent,
    build_control_pack,
    summarize_tool_log,
)
from evopilot.controller import RecoveryController
from evopilot.policy import VerifyPolicy
from evopilot.replan import format_debug_statement, format_improve_suffix
from evopilot.types import AgentSnapshot, ControlDecision, Decision, VerificationEvidence


def boot() -> RecoveryController:
    controller = RecoveryController()
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
        candidate_score=0.10,
        compiled=True,
        regression=True,
        structural_error=True,
        local_error=False,
        resolved_target=False,
        failure_cluster="regression:go",
        change_intent="rewrite the parser",
        targeted_failure_mode="go__poker",
        n_evaluations=10,
        debug_round=0,
        lineage_id=controller.lineage_id,
        resolved_ids=controller.active.resolved_ids,
        unresolved_ids=controller.active.unresolved_ids,
        empty_patch_ids=(),
    )
    base.update(overrides)
    return VerificationEvidence(**base)


def _reply(action: str, reason: str, constraints=None) -> str:
    return json.dumps(
        {
            "action": action,
            "reason": reason,
            "next_constraints": list(constraints or []),
        }
    )


def _install(controller: RecoveryController, action: str, reason: str, constraints=None, sink=None):
    def complete(system, user):
        if sink is not None:
            sink.append(json.loads(user))
        assert "accept" in system and "checkmate" in system
        return _reply(action, reason, constraints)

    bind_control_agent(controller, ControlAgent(complete=complete))


def test_control_prompt_names_the_four_actions_and_forbids_score_overrides():
    for action in ("accept", "debug", "rollback", "checkmate"):
        assert action in CONTROL_AGENT_SYSTEM_PROMPT
    assert "score threshold" in CONTROL_AGENT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "action",
    [Decision.ACCEPT, Decision.DEBUG, Decision.ROLLBACK, Decision.CHECKMATE],
)
def test_mocked_reply_parses_each_action(action):
    agent = ControlAgent(complete=lambda _system, _user: _reply(action.value, f"choose {action.value}", ["stay local"]))
    decision = agent.decide({"verify": {}})
    assert decision == ControlDecision(action, f"choose {action.value}", ("stay local",))


def test_fenced_json_and_one_repair_round():
    calls = {"n": 0}

    def complete(_system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return "I think we should roll back."
        assert "not a valid control decision" in user
        return "```json\n" + _reply("rollback", "the patch never landed", []) + "\n```"

    decision = ControlAgent(complete=complete).decide({"lineage": {}})
    assert decision.action == Decision.ROLLBACK
    assert decision.reason == "the patch never landed"
    assert calls["n"] == 2


def test_a_second_unparseable_reply_is_not_turned_into_a_score_decision(monkeypatch):
    controller = boot()
    bind_control_agent(controller, ControlAgent(complete=lambda _system, _user: "still not json"))

    def explode(*_args, **_kwargs):
        raise AssertionError("policy must not decide")

    monkeypatch.setattr(VerifyPolicy, "decide", explode)
    controller.begin_attempt("rewrite the parser", "go__poker")
    with pytest.raises(ControlAgentParseError):
        controller.step(evidence(controller, candidate_score=0.90, structural_error=False, regression=False))
    assert controller.active.agent_id == "initial"
    assert [agent.agent_id for agent in controller.bank.ordered()] == ["initial"]


def test_accept_is_executed_even_when_the_score_would_roll_back(monkeypatch):
    controller = boot()
    _install(controller, "accept", "keep this direction despite the drop")

    def explode(*_args, **_kwargs):
        raise AssertionError("policy ran")

    monkeypatch.setattr(VerifyPolicy, "decide", explode)
    controller.begin_attempt("rewrite the parser", "go__poker")
    result = controller.step(
        evidence(controller, candidate_id="worse", candidate_score=0.10, structural_error=True)
    )
    assert result.decision == Decision.ACCEPT
    assert result.reason == "keep this direction despite the drop"
    assert result.active.agent_id == "worse"
    assert controller.best_agent_id == "initial"
    assert "worse" in [agent.agent_id for agent in controller.bank.ordered()]
    assert controller.ledger.events[-1]["details"]["decision_source"] == "llm"
    assert controller.ledger.events[-1]["details"]["reason"] != "structural_or_catastrophic_regression"


def test_debug_stays_on_the_candidate_and_keeps_the_agents_constraint():
    controller = boot()
    _install(controller, "debug", "the failure is local", ["keep the parser change small"])
    controller.begin_attempt("rewrite the parser", "go__poker")
    item = evidence(controller, candidate_score=0.39, structural_error=False, local_error=True, regression=True)
    result = controller.step(item)
    assert result.decision == Decision.DEBUG
    assert result.active.agent_id == "initial"
    assert controller._debug_count == 1
    assert controller.memory.open_entry is not None
    assert controller.memory.open_entry.outcome == "open"
    text = format_debug_statement(item, result.replan)
    assert "keep the parser change small" in text
    assert "rewrite the parser" in text


def test_rollback_is_executed_even_when_the_score_would_accept():
    controller = boot()
    _install(controller, "rollback", "the edit is off distribution")
    controller.begin_attempt("rewrite the parser", "go__poker")
    result = controller.step(
        evidence(
            controller,
            candidate_id="better",
            candidate_score=0.90,
            structural_error=False,
            regression=False,
            resolved_target=True,
            failure_cluster="ok",
        )
    )
    assert result.decision == Decision.ROLLBACK
    assert result.reason == "the edit is off distribution"
    assert result.active.agent_id == "initial"
    assert "better" not in [agent.agent_id for agent in controller.bank.ordered()]
    assert controller.memory.open_entry is None


def test_checkmate_switches_lineage_without_a_score_rule():
    controller = boot()
    controller.begin_attempt("cache tool results", "go__poker")
    accepted = controller.step(
        evidence(
            controller,
            candidate_id="c1",
            candidate_score=0.55,
            structural_error=False,
            regression=False,
            resolved_target=True,
            failure_cluster="ok",
            change_intent="cache tool results",
        )
    )
    assert accepted.decision == Decision.ACCEPT
    assert controller.active.agent_id == "c1"

    _install(controller, "checkmate", "this line has no writes left", ["do not rewrite the parser"])
    controller.begin_attempt("rewrite the parser", "cpp__clock")
    stuck_lineage = controller.lineage_id
    result = controller.step(
        evidence(
            controller,
            candidate_id="stuck",
            candidate_score=0.50,
            structural_error=False,
            regression=False,
            failure_cluster="no_gain",
            change_intent="rewrite the parser",
        )
    )
    assert result.decision == Decision.CHECKMATE
    assert result.reason == "this line has no writes left"
    assert result.rollback_mode == "deeper"
    assert result.active.agent_id == "initial"
    assert result.rollback_target.agent_id == "initial"
    assert controller.memory.open_entry is None
    assert controller.memory.entries[-1].outcome == "checkmate"
    assert "rewrite the parser" in controller.memory.recent_failed_intents()
    assert result.replan.broaden_scope is True
    suffix = format_improve_suffix(result.replan)
    assert "do not rewrite the parser" in suffix
    assert "rewrite the parser" in suffix
    controller.begin_attempt("split the tool loop", "cpp__clock")
    assert controller.lineage_id != stuck_lineage
    assert controller.active.agent_id == "initial"


def test_debug_bound_ends_the_lineage_without_rewriting_the_decision():
    controller = boot()
    _install(controller, "debug", "one more local pass")
    controller.begin_attempt("rewrite the parser", "go__poker")
    result = controller.step(evidence(controller, structural_error=False, local_error=True))
    assert result.decision == Decision.DEBUG
    controller.close_bounded_debug()
    assert controller.active.agent_id == "initial"
    assert controller.memory.open_entry is None
    assert controller.attempts[-1].decision == "debug"
    assert "rewrite the parser" in controller.memory.recent_failed_intents()


def test_force_policy_still_uses_the_rule_table():
    controller = boot()
    _install(controller, "accept", "ignore the regression")
    controller.begin_attempt("rewrite the parser", "go__poker")
    result = controller.step(evidence(controller, structural_error=True), force_policy=True)
    assert result.decision == Decision.ROLLBACK
    assert result.reason == "structural_or_catastrophic_regression"
    assert controller.ledger.events[-1]["details"]["decision_source"] == "policy"


def test_evidence_pack_reads_patch_logs_verify_bank_and_lineage(tmp_path):
    root = tmp_path / "output"
    folder = root / "cand"
    folder.mkdir(parents=True)
    (folder / "model_patch.diff").write_text(
        "diff --git a/coding_agent.py b/coding_agent.py\n",
        encoding="utf-8",
    )
    (folder / "self_evo.md").write_text(
        "\n".join(
            [
                "Tool use parse failed: the <tool_use> block was detected but not executed",
                "Tool Used: editor",
                "Tool Input: {'command': 'view', 'path': '/dgm/coding_agent.py', 'view_range': [1, 40]}",
                "Tool Result: 1: class X",
                "Tool Used: editor",
                "Tool Input: {'command': 'edit', 'path': '/dgm/coding_agent.py', 'file_text': 'x'}",
                "Tool Result: ok",
                "Tool Used: bash",
                "Tool Input: {'command': 'pytest'}",
                "Tool Result: failed",
            ]
        ),
        encoding="utf-8",
    )
    (folder / "metadata.json").write_text(
        json.dumps({"evopilot_role": "debug", "model_patch_notempty": True}),
        encoding="utf-8",
    )
    controller = boot()
    controller.begin_attempt("retry a miss", "go__poker")
    controller.step(
        evidence(
            controller,
            candidate_id="old",
            structural_error=True,
            failure_cluster="missed:go__poker",
            change_intent="retry a miss",
        )
    )
    controller.output_dir = str(root)
    controller.workspace_dir = str(folder)
    controller.begin_attempt("rewrite the parser", "cpp__clock")
    item = evidence(
        controller,
        candidate_score=0.25,
        compiled=False,
        structural_error=False,
        local_error=True,
        failure_cluster="compile_or_empty_patch",
        resolved_ids=("python__beer-song",),
        unresolved_ids=("go__poker",),
        empty_patch_ids=("cpp__clock",),
        protocol_mismatch=False,
        score_protocol=("python__beer-song", "go__poker", "cpp__clock"),
    )
    pack = build_control_pack(item, controller, workspace_dir=str(folder), output_dir=str(root))

    assert pack["model_patch"]["present"] is True
    assert pack["model_patch"]["empty"] is False
    assert pack["model_patch"]["files_changed"] == ["coding_agent.py"]
    assert pack["model_patch"]["model_patch_notempty"] is True
    logs = pack["agent_logs"]
    assert logs["tool_parse_failures"] == 1
    assert logs["view_calls"] == 1
    assert logs["write_calls"] == 1
    assert logs["other_tool_calls"] == 1
    assert logs["view_only"] is False
    assert logs["runs"][0]["role"] == "debug"
    assert logs["runs"][0]["log_file"] == "self_evo.md"
    verify = pack["verify"]
    assert verify["candidate_score"] == 0.25
    assert verify["resolved_ids"] == ["python__beer-song"]
    assert verify["unresolved_ids"] == ["go__poker"]
    assert verify["empty_patch_ids"] == ["cpp__clock"]
    assert verify["compiled"] is False
    assert verify["protocol_comparable"] is True
    assert pack["checkpoint_bank"]["active_id"] == "initial"
    assert [member["agent_id"] for member in pack["checkpoint_bank"]["members"]] == ["initial"]
    assert pack["lineage"]["debug_count_used"] == 0
    assert "retry a miss" in pack["lineage"]["recent_failure_intents"]

    empty = folder / "model_patch.diff"
    empty.write_text("\n", encoding="utf-8")
    emptied = build_control_pack(item, controller, workspace_dir=str(folder), output_dir=str(root))
    assert emptied["model_patch"]["empty"] is True
    assert emptied["model_patch"]["files_changed"] == []


def test_view_only_log_has_no_writes():
    summary = summarize_tool_log(
        "Tool Used: editor\nTool Input: {'command': 'view', 'path': '/dgm/a.py'}\nTool Result: ok\n"
    )
    assert summary["view_only"] is True
    assert summary["write_calls"] == 0
    assert summary["tool_parse_failures"] == 0


def test_live_controller_flag_defaults_to_the_agent_and_policy_is_the_off_switch():
    assert parse_args([]).controller == "llm"
    assert parse_args(["--controller", "policy"]).controller == "policy"
    controller = boot()
    install_controller_decider(controller, parse_args(["--controller", "policy"]))
    assert controller.decider is None
    assert controller.decision_source == "policy"
    install_controller_decider(
        controller,
        parse_args(["--controller", "llm", "--control_model", "qwen/qwen3-coder-flash"]),
    )
    assert controller.decider is not None
    assert controller.decision_source == "llm"
    assert callable(EvoPilot_outer.main)


def test_controller_state_roundtrip_keeps_agent_constraints_but_not_the_callable():
    controller = boot()
    controller._agent_constraints = ["keep edits local"]
    controller._broaden_once = True
    controller.decision_source = "llm"
    restored = RecoveryController.from_dict(controller.to_dict())
    assert restored._agent_constraints == ["keep edits local"]
    assert restored._broaden_once is True
    assert restored.decision_source == "llm"
    assert restored.decider is None
