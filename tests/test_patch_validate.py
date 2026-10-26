"""Malformed model_patch.diff is repaired or refused before Polyglot Verify."""

import subprocess
from pathlib import Path

import pytest

from utils.git_utils import (
    dry_run_apply_patch,
    ensure_trailing_newline,
    patch_looks_well_formed,
    preflight_model_patches,
    prepare_model_patch,
    truncate_to_complete_file_diffs,
)


def _file_diff(name: str, old_count: int, body: str) -> str:
    return (
        f"diff --git a/{name} b/{name}\n"
        "index 1111111..2222222 100644\n"
        f"--- a/{name}\n"
        f"+++ b/{name}\n"
        f"@@ -1,{old_count} +1,{old_count} @@\n"
        f"{body}"
    )


def test_missing_trailing_newline_is_repaired_without_dropping_the_diff():
    complete = _file_diff("foo.py", 2, " line1\n-line2\n+line2b\n")
    assert patch_looks_well_formed(complete) == (True, "ok")
    broken = complete[:-1]
    assert patch_looks_well_formed(broken) == (False, "missing_trailing_newline")
    fixed, notes = prepare_model_patch(broken)
    assert notes == {}
    assert fixed == complete
    assert fixed.endswith("\n")


def test_truncated_hunk_is_salvaged_down_to_complete_file_blocks():
    good = _file_diff("foo.py", 2, " line1\n-line2\n+line2b\n")
    # Header asks for 3 old lines; the body was cut after 2.
    tail = _file_diff("bar.py", 3, " line1\n-line2\n")
    raw = good + tail
    assert patch_looks_well_formed(raw)[0] is False
    assert patch_looks_well_formed(raw)[1] == "truncated_hunk"
    fixed, notes = prepare_model_patch(raw)
    assert notes["model_patch_salvaged"] is True
    assert notes["model_patch_malformed_reason"] == "truncated_hunk"
    assert "a/foo.py" in fixed
    assert "a/bar.py" not in fixed
    assert patch_looks_well_formed(fixed) == (True, "ok")
    assert truncate_to_complete_file_diffs(raw) == fixed


def test_mid_line_tail_does_not_start_verify_when_nothing_complete_remains():
    # Header asks for 3 lines. Adding the missing newline still leaves the hunk short.
    chopped = _file_diff("foo.py", 3, " line1\n-line2\n+line2")
    assert not chopped.endswith("\n")
    fixed, notes = prepare_model_patch(chopped)
    assert fixed is None
    assert notes["model_patch_malformed_reason"] == "truncated_hunk"
    assert "model_patch_salvage_failure" in notes


def test_preflight_rejects_a_truncated_parent_and_repairs_a_missing_newline(tmp_path):
    bad = tmp_path / "parent.diff"
    bad.write_text(_file_diff("bar.py", 3, " line1\n-line2\n"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing Verify"):
        preflight_model_patches([str(bad)])

    missing_nl = tmp_path / "newline.diff"
    complete = _file_diff("foo.py", 2, " line1\n-line2\n+line2b\n")
    missing_nl.write_text(complete[:-1], encoding="utf-8")
    preflight_model_patches([str(missing_nl)])
    assert missing_nl.read_text(encoding="utf-8") == complete


def test_git_apply_check_accepts_the_newline_repair(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=repo)
    (repo / "foo.py").write_text("line1\nline2\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "foo.py"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)
    (repo / "foo.py").write_text("line1\nline2b\n", encoding="utf-8")
    diff = subprocess.check_output(["git", "diff"], cwd=repo, text=True)
    assert diff.endswith("\n")
    fixed, notes = prepare_model_patch(diff[:-1])
    assert notes == {}
    assert fixed == ensure_trailing_newline(diff[:-1])
    (repo / "foo.py").write_text("line1\nline2\n", encoding="utf-8")
    ok, reason = dry_run_apply_patch(str(repo), fixed)
    assert ok, reason


def test_callers_validate_before_a_polyglot_worker_pool():
    harness = Path("polyglot/harness.py").read_text(encoding="utf-8")
    function = harness.split("def harness(", 1)[1]
    assert function.index("preflight_model_patches") < function.index("ThreadPoolExecutor(")
    assert "ensure_trailing_newline(remove_patch_by_files" in function

    step = Path("self_improve_step.py").read_text(encoding="utf-8")
    assert step.index("prepare_model_patch") < step.index("run_harness_polyglot")
    assert "model_patch_rejected" in step
    assert "ensure_trailing_newline(diff_versus_commit" in Path("coding_agent.py").read_text(encoding="utf-8")
    assert "ensure_trailing_newline(diff_versus_commit" in Path("coding_agent_polyglot.py").read_text(encoding="utf-8")
    assert "evopilot" not in Path("DGM_outer.py").read_text(encoding="utf-8")
