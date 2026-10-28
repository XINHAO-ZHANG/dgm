"""OpenRouter wiring for the self-improve smoke path. No network calls."""

from pathlib import Path

import llm
import llm_withtools


def test_gpt4o_mini_client_uses_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    client, model = llm.create_client("gpt-4o-mini")
    assert model == "gpt-4o-mini"
    assert not isinstance(client, tuple)
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_gpt_client_accepts_openrouter_key_when_openai_key_is_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    client, model = llm.create_client("gpt-4o-mini")
    assert model == "gpt-4o-mini"
    assert client.api_key == "test-openrouter-key"


def test_llama_openrouter_client_returns_a_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    client, model = llm.create_client("llama3.1-405b")
    assert model == "llama3.1-405b"
    assert not isinstance(client, tuple)
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_qwen3_coder_flash_defaults_to_openrouter(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client, model = llm.create_client("qwen/qwen3-coder-flash")
    assert model == "qwen/qwen3-coder-flash"
    assert not isinstance(client, tuple)
    assert client.api_key == "test-openrouter-key"
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"

    captured = {}

    class _Message:
        content = "ok"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    content, _history = llm.get_response_from_llm(
        "hi",
        _Client(),
        "qwen/qwen3-coder-flash",
        "system",
    )
    assert content == "ok"
    assert captured["model"] == "qwen/qwen3-coder-flash"
    assert captured["messages"][0] == {"role": "system", "content": "system"}


def test_self_improve_model_uses_chat_completions_not_responses(monkeypatch):
    assert llm_withtools.OPENAI_MODEL == "qwen/qwen3-coder-flash"
    called = {}

    def manual(*_args, **_kwargs):
        called["manual"] = True
        return []

    def responses(*_args, **_kwargs):
        called["responses"] = True
        return []

    monkeypatch.setattr(llm_withtools, "chat_with_agent_manualtools", manual)
    monkeypatch.setattr(llm_withtools, "chat_with_agent_openai", responses)
    llm_withtools.chat_with_agent("hi", model=llm_withtools.OPENAI_MODEL)
    assert called == {"manual": True}

    source = Path("self_improve_step.py").read_text(encoding="utf-8")
    assert "diagnose_model = 'qwen/qwen3-coder-flash'" in source
    assert "llm_container_env()" in source
    llm_source = Path("llm.py").read_text(encoding="utf-8")
    assert '"OPENROUTER_API_KEY"' in llm_source
    assert '"OPENAI_BASE_URL"' in llm_source
    assert "def llm_container_env" in llm_source
    for harness_name in ("polyglot/harness.py", "swe_bench/harness.py"):
        harness = Path(harness_name).read_text(encoding="utf-8")
        assert "llm_container_env()" in harness
        assert "resolve_code_model(self_improve=False)" in harness
        assert '"--model"' in harness
    agent = Path("coding_agent.py").read_text(encoding="utf-8")
    polyglot_agent = Path("coding_agent_polyglot.py").read_text(encoding="utf-8")
    assert "resolve_code_model" in agent
    assert "resolve_code_model" in polyglot_agent
    assert "CLAUDE_MODEL if not self_improve else OPENAI_MODEL" not in agent
    assert "CLAUDE_MODEL if not self_improve else OPENAI_MODEL" not in polyglot_agent
    assert "evopilot" not in Path("DGM_outer.py").read_text(encoding="utf-8")


def test_eval_model_defaults_to_openrouter_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("DGM_EVAL_MODEL", raising=False)
    monkeypatch.delenv("DGM_CODE_MODEL", raising=False)
    assert llm.resolve_code_model(self_improve=False) == "qwen/qwen3-coder-flash"
    assert llm.resolve_code_model(self_improve=True) == "qwen/qwen3-coder-flash"
    monkeypatch.setenv("DGM_EVAL_MODEL", "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert llm.resolve_code_model(self_improve=False).startswith("bedrock/")
    assert llm.resolve_code_model(self_improve=True) == "qwen/qwen3-coder-flash"
    assert (
        llm.resolve_code_model(self_improve=False, explicit="qwen/qwen3-coder-flash")
        == "qwen/qwen3-coder-flash"
    )
    assert llm.uses_anthropic_messages("bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert llm.uses_anthropic_messages("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert not llm.uses_anthropic_messages("anthropic/claude-sonnet-4")
    assert not llm.uses_anthropic_messages("qwen/qwen3-coder-flash")


def test_llm_container_env_forwards_openrouter_into_the_sandbox(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OpenRouter_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("DGM_EVAL_MODEL", "qwen/qwen3-coder-flash")
    env = llm.llm_container_env()
    assert env["OPENROUTER_API_KEY"] == "test-openrouter-key"
    assert env["OpenRouter_API_KEY"] == "test-openrouter-key"
    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert env["DGM_EVAL_MODEL"] == "qwen/qwen3-coder-flash"


def test_openrouter_alias_key_is_accepted(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OpenRouter_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client, model = llm.create_client("qwen/qwen3-coder-flash")
    assert model == "qwen/qwen3-coder-flash"
    assert client.api_key == "test-openrouter-key"
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_qwen_client_reports_missing_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OpenRouter_API_KEY", raising=False)
    try:
        llm.create_client("qwen/qwen3-coder-flash")
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OpenRouter key should raise")


def test_chat_helper_logs_llm_failure(monkeypatch, capsys):
    logged = []

    def boom(_model):
        raise RuntimeError("OPENROUTER_API_KEY is unset")

    monkeypatch.setattr(llm_withtools, "load_all_tools", lambda logging=None: [])
    monkeypatch.setattr(llm_withtools, "create_client", boom)
    history = llm_withtools.chat_with_agent_manualtools(
        "hi",
        model="qwen/qwen3-coder-flash",
        logging=logged.append,
    )
    assert history == []
    assert logged
    assert "LLM call failed in chat_with_agent_manualtools" in logged[0]
    assert "OPENROUTER_API_KEY is unset" in logged[0]
    assert "OPENROUTER_API_KEY is unset" in capsys.readouterr().err
