"""Manual <tool_use> blocks must not vanish when parsing fails."""

import ast

import pytest

import llm_withtools

MODEL = "qwen/qwen3-coder-flash"

VALID_EDIT = """<tool_use>
{'tool_name': 'editor', 'tool_input': {'command': 'edit', 'path': '/tmp/a.py', 'file_text': 'print(1)'}}
</tool_use>
"""

JSON_EDIT = """<tool_use>
{"tool_name": "editor", "tool_input": {"command": "edit", "path": "/tmp/a.py", "overwrite": true}}
</tool_use>
"""


def _malformed(payload="unterminated"):
    body = (
        "{'tool_name': 'editor', 'tool_input': {'command': 'edit', "
        "'path': '/tmp/a.py', 'file_text': '" + payload
    )
    return f"<tool_use>\n{body}\n</tool_use>"


def _run_manual(monkeypatch, responses):
    executed = []
    logged = []

    def editor(**kwargs):
        executed.append(kwargs)
        return "wrote"

    seen = []

    def fake_llm(msg, client, model, system_message, print_debug=False, msg_history=None):
        seen.append(msg)
        index = len(seen) - 1
        if index >= len(responses):
            raise AssertionError(f"unexpected LLM call #{len(seen)}: {msg[:180]}")
        text = responses[index]
        history = list(msg_history or [])
        history.append({"role": "assistant", "content": text})
        return text, history

    monkeypatch.setattr(
        llm_withtools,
        "load_all_tools",
        lambda logging=None: [
            {"info": {"name": "editor", "description": "", "input_schema": {}}, "function": editor}
        ],
    )
    monkeypatch.setattr(llm_withtools, "create_client", lambda model: (object(), MODEL))
    monkeypatch.setattr(llm_withtools, "get_response_from_llm", fake_llm)
    history = llm_withtools.chat_with_agent_manualtools(
        "fix it",
        model=MODEL,
        logging=logged.append,
    )
    return history, seen, executed, logged


def test_python_dict_tool_use_matches_literal_eval():
    body = (
        "{'tool_name': 'editor', 'tool_input': "
        "{'command': 'edit', 'path': '/tmp/a.py', 'file_text': 'print(1)'}}"
    )
    parsed = llm_withtools.check_for_tool_use(
        f"<tool_use>\n{body}\n</tool_use>",
        model=MODEL,
    )
    assert parsed == ast.literal_eval(body)


def test_json_bool_tool_use_parses_when_literal_eval_cannot():
    body = '{"tool_name": "editor", "tool_input": {"command": "edit", "overwrite": true}}'
    with pytest.raises(Exception):
        ast.literal_eval(body)
    parsed = llm_withtools.check_for_tool_use(f"<tool_use>{body}</tool_use>", model=MODEL)
    assert parsed["tool_name"] == "editor"
    assert parsed["tool_input"]["overwrite"] is True


def test_missing_tool_input_raises():
    with pytest.raises(llm_withtools.ToolUseParseError, match="tool_input"):
        llm_withtools.check_for_tool_use(
            "<tool_use>{'tool_name': 'editor'}</tool_use>",
            model=MODEL,
        )


def test_no_tool_use_tag_returns_none():
    assert llm_withtools.check_for_tool_use("no tools here", model=MODEL) is None
    assert llm_withtools.check_for_tool_use("see <tool_use> below", model=MODEL) is None


def test_malformed_tool_use_raises():
    with pytest.raises(llm_withtools.ToolUseParseError, match="not executed"):
        llm_withtools.check_for_tool_use(_malformed("broken"), model=MODEL)


def test_valid_tool_use_still_executes(monkeypatch, capsys):
    _history, seen, executed, logged = _run_manual(monkeypatch, [VALID_EDIT, "done"])
    assert executed == [
        {"command": "edit", "path": "/tmp/a.py", "file_text": "print(1)"}
    ]
    assert len(seen) == 2
    assert seen[1].startswith("Tool Used: editor")
    assert not any("Tool use parse failed" in line for line in logged)
    assert "Tool use parse failed" not in capsys.readouterr().err
    assert not any("LLM call failed" in line for line in logged)


def test_json_tool_use_still_executes(monkeypatch):
    _history, seen, executed, logged = _run_manual(monkeypatch, [JSON_EDIT, "done"])
    assert executed == [
        {"command": "edit", "path": "/tmp/a.py", "overwrite": True}
    ]
    assert seen[1].startswith("Tool Used: editor")
    assert not any("Tool use parse failed" in line for line in logged)


def test_malformed_tool_use_is_logged_and_sent_back(monkeypatch, capsys):
    payload = "UNIQUE_BROKEN_FILE_" + ("x" * 5000)
    broken = _malformed(payload)
    _history, seen, executed, logged = _run_manual(
        monkeypatch,
        [broken, "I could not form a tool call."],
    )
    assert executed == []
    assert len(seen) == 2
    assert "Tool use parse failed" in seen[1]
    assert "not executed" in seen[1]
    assert "x" * 200 not in seen[1]
    assert len(seen[1]) < 2000
    assert any("Tool use parse failed" in line for line in logged)
    assert "Tool use parse failed" in capsys.readouterr().err
    assert not any("LLM call failed" in line for line in logged)


def test_malformed_tool_use_stops_after_retry_cap(monkeypatch):
    broken = _malformed("still-broken")
    responses = [broken] * (1 + llm_withtools.MAX_MANUAL_TOOL_PARSE_RETRIES)
    _history, seen, executed, logged = _run_manual(monkeypatch, responses)
    assert executed == []
    assert len(seen) == 1 + llm_withtools.MAX_MANUAL_TOOL_PARSE_RETRIES
    stop_lines = [line for line in logged if "stopping after" in line]
    assert len(stop_lines) == 1
    assert str(llm_withtools.MAX_MANUAL_TOOL_PARSE_RETRIES) in stop_lines[0]


def test_parse_failure_count_resets_after_a_successful_call(monkeypatch):
    monkeypatch.setattr(llm_withtools, "MAX_MANUAL_TOOL_PARSE_RETRIES", 1)
    broken = _malformed("again")
    _history, seen, executed, logged = _run_manual(
        monkeypatch,
        [broken, VALID_EDIT, broken, broken],
    )
    assert executed == [
        {"command": "edit", "path": "/tmp/a.py", "file_text": "print(1)"}
    ]
    assert len(seen) == 4
    assert any("stopping after 1" in line for line in logged)


def test_response_without_tool_use_ends_the_turn(monkeypatch):
    _history, seen, executed, logged = _run_manual(monkeypatch, ["No edit needed."])
    assert executed == []
    assert len(seen) == 1
    assert not any("Tool use parse failed" in line for line in logged)
