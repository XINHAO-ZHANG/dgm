"""Secret values must not survive in logs or saved EvoPilot errors."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from DGM_outer import log_selfimprove_failure
from EvoPilot_outer import failure_metadata
from evopilot.evidence import evidence_from_dgm_metadata
from evopilot.types import AgentSnapshot
from utils.docker_utils import get_thread_logger, safe_log, setup_logger
from utils.secret_redaction import redact_secrets

OPENROUTER = "sk-or-v1-supersecretvalue"
OPENAI = "sk-proj-openaikeyvalue1234"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG"
BASE_URL = "https://openrouter.ai/api/v1"
RAW = (
    f"OPENROUTER_API_KEY={OPENROUTER} "
    f"OPENAI_API_KEY={OPENAI} "
    f"AWS_SECRET_ACCESS_KEY={AWS_SECRET} "
    f"OPENAI_BASE_URL={BASE_URL}"
)


def _assert_redacted(text: str) -> None:
    assert OPENROUTER not in text
    assert OPENAI not in text
    assert AWS_SECRET not in text
    assert f"OPENAI_BASE_URL={BASE_URL}" in text


def test_redact_secrets_keeps_base_url_and_strips_keys():
    redacted = redact_secrets(RAW)
    _assert_redacted(redacted)
    assert "OPENROUTER_API_KEY=[REDACTED]" in redacted
    assert "OPENAI_API_KEY=[REDACTED]" in redacted
    assert "AWS_SECRET_ACCESS_KEY=[REDACTED]" in redacted
    assert redact_secrets(redacted) == redacted

    bare = redact_secrets(f"token {OPENROUTER} and {OPENAI} url {BASE_URL}")
    assert OPENROUTER not in bare
    assert OPENAI not in bare
    assert BASE_URL in bare

    quoted = redact_secrets(
        "{'OPENAI_API_KEY': '" + OPENAI + "', 'OPENAI_BASE_URL': '" + BASE_URL + "'}"
    )
    assert OPENAI not in quoted
    assert BASE_URL in quoted


def test_redact_timeout_exception_string():
    exc = subprocess.TimeoutExpired(
        [
            "apptainer",
            "exec",
            "--env",
            f"OPENROUTER_API_KEY={OPENROUTER}",
            "--env",
            f"OPENAI_API_KEY={OPENAI}",
            "--env",
            f"AWS_SECRET_ACCESS_KEY={AWS_SECRET}",
            "--env",
            f"OPENAI_BASE_URL={BASE_URL}",
        ],
        30,
    )
    blob = redact_secrets(str(exc) + "\n" + repr(exc))
    _assert_redacted(blob)


def _read_log(logger, path: Path) -> str:
    for handler in logger.handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_dgm_outer_failure_log_redacts_secrets(tmp_path):
    log_path = tmp_path / "dgm_outer.log"
    logger = setup_logger(str(log_path))
    try:
        raise subprocess.TimeoutExpired(
            ["apptainer", "--env", f"OPENROUTER_API_KEY={OPENROUTER}", "--env", f"AWS_SECRET_ACCESS_KEY={AWS_SECRET}", "--env", f"OPENAI_BASE_URL={BASE_URL}"],
            30,
        )
    except subprocess.TimeoutExpired as exc:
        log_selfimprove_failure(logger, exc)
    text = _read_log(logger, log_path)
    assert "Self-improvement step failed" in text
    assert "Traceback" in text
    _assert_redacted(text)


def test_dgm_logger_redacts_a_raw_error_string(tmp_path):
    log_path = tmp_path / "dgm_outer.log"
    logger = setup_logger(str(log_path))
    logger.error("Self-improvement step failed: %s", RAW)
    text = _read_log(logger, log_path)
    _assert_redacted(text)


def test_safe_log_redacts_secrets(tmp_path):
    log_path = tmp_path / "selfimprove.log"
    setup_logger(str(log_path))
    safe_log(f"container failed: {RAW}")
    text = _read_log(get_thread_logger(), log_path)
    _assert_redacted(text)


def test_swe_safe_log_redacts_secrets(tmp_path):
    pytest.importorskip("docker")
    from swe_bench.utils import safe_log as swe_safe_log
    from swe_bench.utils import setup_logger as swe_setup

    log_path = tmp_path / "swe.log"
    logger = swe_setup(str(log_path))
    swe_safe_log(f"harness failed: {RAW}")
    text = _read_log(logger, log_path)
    _assert_redacted(text)


def test_evopilot_saved_error_redacts_secrets():
    try:
        raise subprocess.CalledProcessError(
            1,
            [
                "apptainer",
                "--env",
                f"OPENAI_API_KEY={OPENAI}",
                "--env",
                f"OPENROUTER_API_KEY={OPENROUTER}",
                "--env",
                f"OPENAI_BASE_URL={BASE_URL}",
            ],
        )
    except subprocess.CalledProcessError as exc:
        metadata = failure_metadata("initial", "go__poker", exc)
    blob = json.dumps(metadata)
    assert OPENAI not in blob
    assert OPENROUTER not in blob
    assert BASE_URL in blob
    assert metadata["evopilot_failure_cluster"] == "infrastructure_error"

    base = AgentSnapshot(agent_id="initial", parent_id=None, score=0.4, step=0, root=True)
    evidence = evidence_from_dgm_metadata(
        {
            "run_id": "failed-1",
            "entry": "go__poker",
            "model_patch_notempty": False,
            "error": RAW,
            "overall_performance": {},
        },
        base=base,
    )
    _assert_redacted(evidence.notes)
