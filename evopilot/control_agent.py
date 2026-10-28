"""LLM Control Agent for the decision that used to live in ``VerifyPolicy``.

After Improve and Verify, the agent reads a workspace evidence pack built
from artifacts the run already wrote and returns ``accept``, ``debug``,
``rollback``, or ``checkmate``. The outer loop executes that action. Score
thresholds stay inside ``VerifyPolicy``, which is the off switch.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from evopilot.controller import RecoveryController
from evopilot.evidence import parse_diff_paths
from evopilot.types import ControlDecision, Decision, VerificationEvidence
from utils.secret_redaction import redact_secrets

CONTROL_ACTIONS = {
    "accept": Decision.ACCEPT,
    "debug": Decision.DEBUG,
    "rollback": Decision.ROLLBACK,
    "checkmate": Decision.CHECKMATE,
}

CONTROL_AGENT_SYSTEM_PROMPT = """You are the EvoPilot Control Agent.

You run after Improve and Verify. You read one workspace evidence pack and choose the next action. The outer loop executes your action. It does not replace that action with a score threshold, an accept margin, or a catastrophic-drop rule.

Choose exactly one action:
- accept: commit this candidate into the checkpoint bank and make it the active agent. Do not choose accept when candidate_score is null.
- debug: repair this candidate in place, then verify again. Choose this only when the evidence still looks locally repairable and debug_count_used is below debug_budget.
- rollback: abandon this candidate and continue from a checkpoint already in the bank. The controller selects the bank target.
- checkmate: this lineage is exhausted. End it and switch lineage. The controller uses its existing escalation path: a deeper checkpoint when one exists, then a new improve with a broader replan.

Use only the evidence pack. Tool-parse failures, view-only traces, empty patches, compile or structure errors, protocol mismatch, the checkpoint bank, the debug count, and recent failure intents are inputs. Do not invent missing measurements. If protocol_comparable is false, do not treat the scores as comparable.

Return one JSON object and no other text:
{"action": "accept" | "debug" | "rollback" | "checkmate", "reason": "<one short paragraph>", "next_constraints": ["<optional falsifiable constraint>"]}

next_constraints may be an empty list. Each constraint must be something a later edit can obey or falsify.
"""

_TOOL_USED = re.compile(r"^Tool Used:\s*([A-Za-z0-9_\-]+)\s*$")
_COMMAND = re.compile(r"""['"]command['"]\s*:\s*['"](view|create|edit)['"]""")
_PARSE_FAIL = "Tool use parse failed"
_LOG_LIMIT = 1_000_000


class ControlAgentParseError(ValueError):
    """The model reply was not a control decision. This is not a score rule."""


CompleteFn = Callable[[str, str], str]


def default_complete(system: str, user: str, model: str) -> str:
    """One chat completion on the Improve backbone (OpenRouter qwen by default)."""

    from llm import create_client, get_response_from_llm

    client, client_model = create_client(model)
    content, _history = get_response_from_llm(
        msg=user,
        client=client,
        model=client_model,
        system_message=system,
        print_debug=False,
        msg_history=[],
        temperature=0.0,
    )
    if content is None:
        return ""
    return str(content)


class ControlAgent:
    def __init__(self, model: str | None = None, complete: CompleteFn | None = None) -> None:
        from llm import DEFAULT_CODE_MODEL

        self.model = model or DEFAULT_CODE_MODEL
        self._complete = complete

    def decide(self, pack: dict) -> ControlDecision:
        user = json.dumps(pack, indent=2, sort_keys=True, default=str)
        raw = self._call(CONTROL_AGENT_SYSTEM_PROMPT, user)
        try:
            return parse_control_decision(raw)
        except ControlAgentParseError as exc:
            repair = (
                user
                + "\n\nThe previous reply was not a valid control decision.\n"
                + f"Error: {exc}\n"
                + "Reply again with one JSON object only. "
                + 'Keys are "action", "reason", and "next_constraints".'
            )
            return parse_control_decision(self._call(CONTROL_AGENT_SYSTEM_PROMPT, repair))

    def _call(self, system: str, user: str) -> str:
        if self._complete is not None:
            return self._complete(system, user)
        return default_complete(system, user, self.model)


def parse_control_decision(text: str) -> ControlDecision:
    payload = _extract_json_object(text or "")
    action_name = str(payload.get("action", "")).strip().lower()
    if action_name not in CONTROL_ACTIONS:
        raise ControlAgentParseError(
            f"action must be one of {', '.join(CONTROL_ACTIONS)}, got {action_name!r}"
        )
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ControlAgentParseError("reason is empty")
    constraints = _constraints(payload.get("next_constraints", []))
    return ControlDecision(
        action=CONTROL_ACTIONS[action_name],
        reason=reason,
        next_constraints=constraints,
    )


def bind_control_agent(controller: RecoveryController, agent: ControlAgent) -> None:
    """Install ``agent`` as the only post-verify decider on ``controller``."""

    def decide(evidence: VerificationEvidence) -> ControlDecision:
        pack = build_control_pack(
            evidence,
            controller,
            workspace_dir=getattr(controller, "workspace_dir", None),
            output_dir=getattr(controller, "output_dir", None),
        )
        return agent.decide(pack)

    controller.decider = decide
    controller.decision_source = "llm"


def build_control_pack(
    evidence: VerificationEvidence,
    controller: RecoveryController,
    *,
    workspace_dir: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Assemble the minimum evidence pack from the controller and run files.

    Missing files stay null. Nothing here is a score the caller invents.
    """

    folders = _lineage_folders(evidence, controller, workspace_dir, output_dir)
    current = workspace_dir if workspace_dir and os.path.isdir(workspace_dir) else None
    if current is None and folders:
        current = folders[-1]
    return {
        "model_patch": _model_patch_section(current, evidence),
        "agent_logs": _agent_logs(folders),
        "verify": _verify_section(evidence),
        "checkpoint_bank": _bank_section(controller),
        "lineage": _lineage_section(evidence, controller),
    }


def summarize_tool_log(text: str) -> dict:
    """Count tool-parse failures and view calls versus writes in an agent log."""

    view_calls = 0
    write_calls = 0
    other_calls = 0
    samples: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _PARSE_FAIL in line and len(samples) < 3:
            samples.append(redact_secrets(line.strip())[:240])
        match = _TOOL_USED.match(line.strip())
        if match is None:
            continue
        name = match.group(1)
        window = "\n".join(lines[index : index + 8])
        if name == "editor":
            command = _COMMAND.search(window)
            if command is None:
                other_calls += 1
            elif command.group(1) == "view":
                view_calls += 1
            else:
                write_calls += 1
        else:
            other_calls += 1
    return {
        "tool_parse_failures": text.count(_PARSE_FAIL),
        "view_calls": view_calls,
        "write_calls": write_calls,
        "other_tool_calls": other_calls,
        "view_only": view_calls > 0 and write_calls == 0 and other_calls == 0,
        "parse_failure_samples": samples,
    }


def _constraints(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = " ".join(value.split())
        return (text,) if text else ()
    if not isinstance(value, list):
        raise ControlAgentParseError("next_constraints must be a list of strings")
    items: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text and text not in items:
            items.append(text)
    return tuple(items)


def _extract_json_object(text: str) -> dict:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            parsed, _end = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
    raise ControlAgentParseError("no JSON object with an action field")


def _lineage_folders(
    evidence: VerificationEvidence,
    controller: RecoveryController,
    workspace_dir: str | None,
    output_dir: str | None,
) -> list[str]:
    found: list[str] = []

    def add(path: str | None) -> None:
        if not path:
            return
        folder = os.path.abspath(path)
        if folder not in found and os.path.isdir(folder):
            found.append(folder)

    add(workspace_dir)
    root = output_dir or getattr(controller, "output_dir", None)
    if not root:
        return found
    ids: list[str] = []
    open_entry = controller.memory.open_entry
    if open_entry is not None:
        ids.extend(open_entry.candidate_ids)
    if evidence.candidate_id:
        ids.append(str(evidence.candidate_id))
    for candidate_id in ids:
        if candidate_id and not str(candidate_id).startswith("failed"):
            add(os.path.join(root, str(candidate_id)))
    return found


def _model_patch_section(folder: str | None, evidence: VerificationEvidence) -> dict:
    files = list(evidence.files_or_mechanisms)
    section: dict = {
        "present": None,
        "empty": False if files else None,
        "files_changed": files,
        "model_patch_notempty": None,
        "model_patch_rejected": None,
    }
    if folder is None:
        return section
    metadata = _read_json(os.path.join(folder, "metadata.json"))
    if "model_patch_notempty" in metadata:
        section["model_patch_notempty"] = bool(metadata.get("model_patch_notempty"))
    if "model_patch_rejected" in metadata:
        section["model_patch_rejected"] = bool(metadata.get("model_patch_rejected"))
    path = os.path.join(folder, "model_patch.diff")
    if not os.path.isfile(path):
        section["present"] = False
        if section["empty"] is None:
            section["empty"] = True
        return section
    text = _read_text(path)
    if text is None:
        section["present"] = True
        return section
    changed = list(parse_diff_paths(text))
    section["present"] = True
    section["empty"] = len(text.strip()) == 0
    section["files_changed"] = changed or files
    return section


def _agent_logs(folders: list[str]) -> dict:
    runs = []
    for folder in folders:
        text, source = _agent_log_text(folder)
        summary = summarize_tool_log(text)
        metadata = _read_json(os.path.join(folder, "metadata.json"))
        role = metadata.get("evopilot_role") if isinstance(metadata.get("evopilot_role"), str) else None
        runs.append(
            {
                "run_id": os.path.basename(folder),
                "role": role,
                "log_file": source,
                **summary,
            }
        )
    return {
        "tool_parse_failures": sum(item["tool_parse_failures"] for item in runs),
        "view_calls": sum(item["view_calls"] for item in runs),
        "write_calls": sum(item["write_calls"] for item in runs),
        "other_tool_calls": sum(item["other_tool_calls"] for item in runs),
        "view_only": any(item["view_calls"] for item in runs)
        and all(item["write_calls"] == 0 and item["other_tool_calls"] == 0 for item in runs),
        "runs": runs,
    }


def _verify_section(evidence: VerificationEvidence) -> dict:
    return {
        "candidate_id": evidence.candidate_id,
        "candidate_score": evidence.candidate_score,
        "base_score": evidence.base_score,
        "compiled": evidence.compiled,
        "structural_error": evidence.structural_error,
        "local_error": evidence.local_error,
        "regression": evidence.regression,
        "protocol_comparable": not evidence.protocol_mismatch,
        "protocol_mismatch": evidence.protocol_mismatch,
        "score_protocol": list(evidence.score_protocol),
        "resolved_ids": list(evidence.resolved_ids),
        "unresolved_ids": list(evidence.unresolved_ids),
        "empty_patch_ids": list(evidence.empty_patch_ids),
        "failure_cluster": evidence.failure_cluster,
        "resolved_target": evidence.resolved_target,
        "notes": evidence.notes,
    }


def _bank_section(controller: RecoveryController) -> dict:
    active = controller.active
    members = []
    for agent in controller.bank.ordered():
        members.append(
            {
                "agent_id": agent.agent_id,
                "parent_id": agent.parent_id,
                "score": agent.score,
                "step": agent.step,
                "root": agent.root,
                "active": active is not None and agent.agent_id == active.agent_id,
            }
        )
    return {
        "active_id": None if active is None else active.agent_id,
        "active_score": None if active is None else active.score,
        "members": members,
    }


def _lineage_section(evidence: VerificationEvidence, controller: RecoveryController) -> dict:
    return {
        "lineage_id": evidence.lineage_id or controller.lineage_id,
        "debug_count_used": controller._debug_count,
        "debug_round": evidence.debug_round,
        "debug_budget": controller.policy.debug_budget,
        "change_intent": evidence.change_intent,
        "targeted_failure_mode": evidence.targeted_failure_mode,
        "recent_failure_intents": controller.memory.recent_failed_intents(),
    }


def _agent_log_text(folder: str) -> tuple[str, str | None]:
    evo_path = os.path.join(folder, "self_evo.md")
    improve_path = os.path.join(folder, "self_improve.log")
    evo = _read_text(evo_path)
    improve = _read_text(improve_path)
    evo_markers = _has_tool_markers(evo)
    improve_markers = _has_tool_markers(improve)
    improve_has_parse_failure = bool(improve) and _PARSE_FAIL in improve
    evo_missing_parse_failure = evo is None or _PARSE_FAIL not in evo
    if evo_markers and improve_markers and improve_has_parse_failure and evo_missing_parse_failure:
        return f"{evo or ''}\n{improve}", "self_evo.md+self_improve.log"
    if evo_markers:
        return evo or "", "self_evo.md"
    if improve_markers:
        return improve or "", "self_improve.log"
    if evo is not None:
        return evo, "self_evo.md"
    if improve is not None:
        return improve, "self_improve.log"
    return "", None


def _has_tool_markers(text: str | None) -> bool:
    return bool(text) and ("Tool Used:" in text or _PARSE_FAIL in text)


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(_LOG_LIMIT)
    except OSError:
        return None


def _read_json(path: str) -> dict:
    text = _read_text(path)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
