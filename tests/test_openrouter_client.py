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


def test_self_improve_model_uses_chat_completions_not_responses(monkeypatch):
    assert llm_withtools.OPENAI_MODEL == "gpt-4o-mini"
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
    assert "diagnose_model = 'gpt-4o-mini'" in source
    assert '"OPENROUTER_API_KEY"' in source
    assert '"OPENAI_BASE_URL"' in source
    agent = Path("coding_agent.py").read_text(encoding="utf-8")
    assert "CLAUDE_MODEL if not self_improve else OPENAI_MODEL" in agent
    assert "evopilot" not in Path("DGM_outer.py").read_text(encoding="utf-8")
