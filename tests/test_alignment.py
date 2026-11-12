"""The DGM baseline aligned with EvoPilot: runtime sync, initial cache, split screen.

No container, model, or network is used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import DGM_outer
import self_improve_step

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import build_initial_from_baseline as initial_builder  # noqa: E402
import sync_runtime  # noqa: E402

BUCKETS = [["go__a", "py__b"], ["rs__c", "go__d"], ["py__e"]]
HELDOUT = ["go__h"]
POOL = ["py__p"]


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def fake_evopilot(tmp_path):
    source = tmp_path / "EvoPilot"
    for path in sync_runtime.RUNTIME_FILES:
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {path}\n", encoding="utf-8")
    git(source, "init", "-q")
    git(source, "add", "-A")
    git(source, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "runtime")
    return source


def test_sync_copies_the_runtime_and_records_the_commit(tmp_path):
    source = fake_evopilot(tmp_path)
    root = tmp_path / "dgm"
    root.mkdir()
    record = sync_runtime.sync(str(source), str(root))
    assert set(record["files"]) == set(sync_runtime.RUNTIME_FILES) and len(record["source_commit"]) == 40
    assert (root / "llm_withtools.py").read_text(encoding="utf-8") == "# llm_withtools.py\n"
    assert sync_runtime.runtime_mismatches(str(root)) == []
    (root / "polyglot" / "harness.py").write_text("changed\n", encoding="utf-8")
    assert sync_runtime.runtime_mismatches(str(root)) == ["polyglot/harness.py"]
    assert "DGM_outer.py" not in sync_runtime.RUNTIME_FILES and "self_improve_step.py" not in sync_runtime.RUNTIME_FILES
    assert "prompts/self_improvement_prompt.py" not in sync_runtime.RUNTIME_FILES


def test_sync_refuses_uncommitted_files_unless_told_and_then_the_check_fails(tmp_path):
    source = fake_evopilot(tmp_path)
    (source / ".dockerignore").write_text("changed\n", encoding="utf-8")
    root = tmp_path / "dgm"
    root.mkdir()
    with pytest.raises(SystemExit, match="Commit these"):
        sync_runtime.sync(str(source), str(root))
    record = sync_runtime.sync(str(source), str(root), allow_uncommitted=True)
    assert record["source_uncommitted"] == [".dockerignore"]
    assert any("uncommitted" in item for item in sync_runtime.runtime_mismatches(str(root)))
    assert sync_runtime.runtime_mismatches(str(tmp_path / "nowhere"))[0].startswith("EVOPILOT_RUNTIME.json")


def test_self_improve_image_keeps_only_the_agent(tmp_path):
    repo = tmp_path / "repo"
    for path in (
        "coding_agent_polyglot.py", "coding_agent.py", "llm.py", "tools/edit.py", "utils/eval_utils.py",
        "utils/apptainer_compat.py", "DGM_outer.py", "evopilot/controller.py", "tests/test_x.py",
        ".claude/settings.local.json", "polyglot/harness.py", "prompts/self_improvement_prompt.py",
    ):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("", encoding="utf-8")
    with open(os.path.join(sync_runtime.ROOT, ".dockerignore"), encoding="utf-8") as handle:
        (repo / ".dockerignore").write_text(handle.read(), encoding="utf-8")
    assert sync_runtime.context_files(str(repo)) == {"coding_agent.py", "llm.py", "tools/edit.py", "utils/apptainer_compat.py"}


def test_self_improve_image_has_its_own_tag():
    assert self_improve_step.SELF_IMPROVE_IMAGE not in ("dgm", "")


def write_calibration(tmp_path):
    calib = tmp_path / "calib"
    per_task = {
        "go__a": ["resolved", "unresolved"],  # resolved: half its runs passed
        "py__b": ["unresolved", "empty_patch", "resolved"],
        "rs__c": ["resolved"],
        "go__d": ["unresolved"],
        "py__e": ["resolved", "resolved"],
        "go__h": ["unresolved"],
        "py__p": ["unresolved"],
    }
    calib.mkdir()
    (calib / "calibration_summary.json").write_text(json.dumps({"per_task": {i: {"outcomes": o} for i, o in per_task.items()}}))
    for rep, results in (("rep_0", {"py__b": "unresolved", "go__d": "unresolved", "go__h": "unresolved"}), ("rep_1", {"py__b": "empty_patch"})):
        folder = calib / rep / "predictions" / f"calib_{rep}_0"
        folder.mkdir(parents=True)
        for instance_id, result in results.items():
            (folder / f"{instance_id}.md").write_text(f"log {instance_id} {rep}", encoding="utf-8")
            (folder / f"{instance_id}.json").write_text(json.dumps({"instance_id": instance_id, "eval_result": result, "model_patch": ""}))
            (folder / f"{instance_id}_eval.md").write_text(f"tests {instance_id} {rep}", encoding="utf-8")
    return calib


def write_split(tmp_path, buckets=BUCKETS):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"eval_buckets": buckets, "heldout": HELDOUT, "diagnosis_pool": POOL}))
    return split


def test_initial_cache_comes_from_the_baseline_on_the_evaluation_buckets(tmp_path):
    calib = write_calibration(tmp_path)
    split = write_split(tmp_path)
    out = tmp_path / "run" / "initial"
    metadata = initial_builder.build(str(split), [str(calib)], str(out))
    small, deep = metadata["overall_performance"], metadata["overall_performance_deep"]
    assert small["accuracy_score"] == pytest.approx(2 / 5) and small["total_resolved_ids"] == ["go__a"]
    assert small["total_unresolved_ids"] == ["py__b"] and small["total_submitted_instances"] == 2
    assert deep["total_submitted_instances"] == 5 and set(deep["total_unresolved_ids"]) == {"py__b", "go__d"}
    assert "go__h" not in metadata["baseline_runs"] and "py__p" not in metadata["baseline_runs"]
    copied = {path.name for path in (out / "predictions").rglob("*.md")}
    assert "go__h.md" not in copied and {"py__b.md", "go__d.md"} <= copied
    # DGM's diagnosis reads one failing run of the problem.
    from prompts.self_improvement_prompt import find_selfimprove_eval_logs

    md_logs, eval_logs, _patches, results = find_selfimprove_eval_logs("py__b", str(tmp_path / "run"), commit_id="initial")
    assert results == ["unresolved"] and md_logs[0] == "log py__b rep_0"
    with pytest.raises(SystemExit, match="written once"):
        initial_builder.build(str(split), [str(calib)], str(out))


def test_initial_cache_refuses_a_problem_it_cannot_diagnose(tmp_path):
    calib = write_calibration(tmp_path)
    split = write_split(tmp_path, [["go__a", "rs__c", "go__d"], ["py__e"]])
    for path in (calib / "rep_0").rglob("go__d.md"):
        path.unlink()
    with pytest.raises(SystemExit, match="go__d"):
        initial_builder.build(str(split), [str(calib)], str(tmp_path / "out"))


def test_split_screen_is_bucket_one_then_the_rest(tmp_path):
    first, rest = DGM_outer.load_split_screen(str(write_split(tmp_path)))
    assert first == ["go__a", "py__b"] and rest == ["rs__c", "go__d", "py__e"]
    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps({"eval_buckets": [["go__a"], ["go__h"]], "heldout": HELDOUT}))
    with pytest.raises(RuntimeError, match="overlap"):
        DGM_outer.load_split_screen(str(leaky))


def test_alignment_check_needs_the_synced_runtime_and_an_initial_cache_of_the_same_split(tmp_path, monkeypatch):
    calib = write_calibration(tmp_path)
    split = write_split(tmp_path)
    initial = tmp_path / "initial_polyglot_qwen"
    monkeypatch.setattr(sync_runtime, "runtime_mismatches", lambda root=None: [])
    with pytest.raises(RuntimeError, match="missing"):
        DGM_outer.check_alignment(str(split), str(initial))
    initial_builder.build(str(split), [str(calib)], str(initial))
    DGM_outer.check_alignment(str(split), str(initial))
    (tmp_path / "other").mkdir()
    other = write_split(tmp_path / "other", [["go__a"], ["py__b"]])
    with pytest.raises(RuntimeError, match="another split"):
        DGM_outer.check_alignment(str(other), str(initial))
    monkeypatch.setattr(sync_runtime, "runtime_mismatches", lambda root=None: ["polyglot/harness.py"])
    with pytest.raises(RuntimeError, match="polyglot/harness.py"):
        DGM_outer.check_alignment(str(split), str(initial))


def test_dgm_under_the_split_scores_children_on_the_buckets(monkeypatch, tmp_path):
    calib = write_calibration(tmp_path)
    split = write_split(tmp_path)
    initial = tmp_path / "initial_polyglot_qwen"
    initial_builder.build(str(split), [str(calib)], str(initial))
    monkeypatch.setattr(sync_runtime, "runtime_mismatches", lambda root=None: [])
    monkeypatch.setenv("DGM_OUTPUT_ROOT", str(tmp_path / "output_dgm"))
    monkeypatch.setattr(DGM_outer, "require_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(DGM_outer, "cleanup_owned_workspaces", lambda *args, **kwargs: [])
    monkeypatch.setattr(DGM_outer, "load_json_file", lambda path: {"source_commit": "abc"} if path == "EVOPILOT_RUNTIME.json" else json.load(open(path)))
    calls = []

    def improve(**kwargs):
        calls.append(kwargs)
        folder = os.path.join(kwargs["output_dir"], f"child{len(calls)}")
        os.makedirs(folder, exist_ok=True)
        metadata = {"run_id": f"child{len(calls)}", "parent_commit": kwargs["parent_commit"]}
        with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        return metadata

    monkeypatch.setattr("self_improve_step.self_improve", improve)
    code = DGM_outer.main([
        "--polyglot", "--no_full_eval", "--selfimprove_size", "1", "--selfimprove_workers", "1", "--max_generation", "1",
        "--split", str(split), "--initial_dir", str(initial),
    ])
    assert code == 0 and len(calls) == 1
    call = calls[0]
    assert call["test_task_list"] == ["go__a", "py__b"] and call["test_task_list_more"] == ["rs__c", "go__d", "py__e"]
    # The first diagnosis reads one of the initial agent's unresolved bucket-1 problems.
    assert call["parent_commit"] == "initial" and call["entry"] == "py__b"
    copied = list((tmp_path / "output_dgm").glob("*/initial/metadata.json"))
    assert copied and json.loads(copied[0].read_text())["overall_performance"]["total_unresolved_ids"] == ["py__b"]
