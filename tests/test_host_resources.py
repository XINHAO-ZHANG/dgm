"""Host resource gates, workspace ownership, and outer-loop fail-fast."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import DGM_outer
import EvoPilot_outer
from EvoPilot_outer import main, parse_args, run_live
from evopilot.types import AgentSnapshot
from utils.apptainer_compat import ApptainerClient
from utils.host_resources import (
    DISK_CODE,
    INFRA_APPTAINER,
    INFRA_LDAP,
    INFRA_MEMORY,
    INFRA_NOSPACE,
    PRESSURE_CODE,
    ResourceGateError,
    ResourceSnapshot,
    apply_infra_metadata,
    assess_resources,
    classify_infra_error,
    cleanup_owned_workspaces,
    raise_for_infra_results,
    stamp_workspace,
    stop_reason_for_infra,
    thresholds_from_args,
    verify_worker_block,
)


_GIB = 1024 ** 3


def _snapshot(*, free=_GIB, mem=8 * _GIB, pressure=0.0, paths=None):
    return ResourceSnapshot(
        paths=paths if paths is not None else {"/tmp": free, "/tmp/apptainer-workspaces": free},
        mem_available_bytes=mem,
        psi_full_avg10=pressure,
        psi_some_avg10=pressure,
    )


def test_low_tmp_and_pressure_are_distinct_from_a_bad_patch():
    thresholds = thresholds_from_args(parse_args([]))
    code, detail = assess_resources(thresholds, _snapshot(free=1024))
    assert code == DISK_CODE
    assert "/tmp" in detail
    assert "10.00 GiB" in detail

    code, detail = assess_resources(thresholds, _snapshot(free=20 * _GIB, mem=512 * 1024 ** 2))
    assert code == "resource_exhausted_memory"
    assert "MemAvailable" in detail

    code, _detail = assess_resources(thresholds, _snapshot(free=20 * _GIB, pressure=40.0))
    assert code == PRESSURE_CODE

    healthy, detail = assess_resources(thresholds, _snapshot(free=20 * _GIB))
    assert healthy is None
    assert detail == "ok"

    disabled = thresholds_from_args(parse_args(["--min-tmp-free-gb", "0", "--min-available-mem-gb", "0", "--max-mem-pressure", "-1"]))
    assert assess_resources(disabled, _snapshot(free=1, mem=1, pressure=99))[0] is None


def test_classify_infra_errors_and_leave_algorithm_failures_alone():
    assert classify_infra_error("OSError: [Errno 28] No space left on device") == INFRA_NOSPACE
    assert classify_infra_error("Cannot allocate memory") == INFRA_MEMORY
    assert classify_infra_error("FATAL: failed to get user id: user: unknown userid 1000") == INFRA_LDAP
    assert classify_infra_error("apptainer sandbox build failed: fakeroot") == INFRA_APPTAINER
    assert classify_infra_error("Model patch file is empty or does not exist") is None
    assert classify_infra_error("eval_result unresolved") is None
    assert stop_reason_for_infra([{"run_id": "child", "model_patch_notempty": False}], continue_on_infra=False) is None
    assert stop_reason_for_infra([{"infra_error_code": INFRA_NOSPACE}], continue_on_infra=False) == INFRA_NOSPACE
    assert stop_reason_for_infra([{"infra_error_code": INFRA_NOSPACE}], continue_on_infra=True) is None

    metadata = apply_infra_metadata({}, INFRA_NOSPACE, "OPENAI_API_KEY=sk-proj-openaikeyvalue1234 No space left on device")
    assert metadata["infra_error_code"] == INFRA_NOSPACE
    assert metadata["evopilot_failure_cluster"] == "infrastructure_error"
    assert "sk-proj-openaikeyvalue1234" not in json.dumps(metadata)
    assert "No space left on device" in metadata["error"]


def test_cleanup_removes_only_this_runs_workspaces(tmp_path, monkeypatch):
    root = tmp_path / "apptainer-workspaces"
    root.mkdir()
    monkeypatch.setenv("APPTAINER_WORKSPACE_ROOT", str(root))
    monkeypatch.setenv("APPTAINER_IMAGE_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("DGM_WORKSPACE_OWNER", "dgm-run-a")
    client = ApptainerClient(timeout=5)
    container = client.containers.create(image="alpine:latest", name="dgm-container-abc")
    owned = container._workspace
    marker = (owned / ".dgm-workspace-owner").read_text(encoding="utf-8")
    assert "owner=dgm-run-a" in marker
    assert "sk-" not in marker

    foreign = root / "pb.eval.other.user"
    foreign.mkdir()
    (foreign / ".dgm-workspace-owner").write_text("owner=someone-else\nname=pb\n", encoding="utf-8")
    unmarked = root / "unmarked-sandbox"
    unmarked.mkdir()
    (unmarked / "rootfs").mkdir()
    link = root / "linked"
    link.symlink_to(foreign, target_is_directory=True)

    removed = cleanup_owned_workspaces(root, owner="dgm-run-a")
    assert not owned.exists()
    assert any(Path(path).name == owned.name for path in removed)
    assert foreign.exists()
    assert unmarked.exists()
    assert link.is_symlink()

    # A second run must not sweep the first run's surviving directory.
    assert cleanup_owned_workspaces(root, owner="dgm-run-b") == []
    assert foreign.exists()


def test_verify_worker_block_does_not_look_like_an_empty_patch(monkeypatch):
    def deny(*, phase):
        raise ResourceGateError(DISK_CODE, "refusing verify: /tmp has 1 MiB free", None)

    monkeypatch.setattr("utils.host_resources.require_resources_from_env", deny)
    blocked = verify_worker_block("go__poker")
    assert blocked["success"] is False
    assert blocked["eval_result"] == "incomplete"
    assert blocked["infra_error_code"] == DISK_CODE
    with pytest.raises(Exception) as caught:
        raise_for_infra_results([blocked])
    assert caught.value.code == DISK_CODE


def test_self_improve_refuses_api_and_sandbox_when_tmp_is_low(monkeypatch, tmp_path):
    from self_improve_step import self_improve

    def deny(*, phase):
        raise ResourceGateError(DISK_CODE, "refusing improve: /tmp has 1 MiB free", None)

    monkeypatch.setattr("self_improve_step.require_resources_from_env", deny)

    def forbid_docker(*_args, **_kwargs):
        raise AssertionError("docker client should not be created")

    def forbid_diagnose(*_args, **_kwargs):
        raise AssertionError("diagnose should not call the model")

    monkeypatch.setattr("self_improve_step.docker_from_env", forbid_docker)
    monkeypatch.setattr("self_improve_step.diagnose_problem", forbid_diagnose)
    metadata = self_improve(
        parent_commit="initial",
        output_dir=str(tmp_path),
        entry="go__poker",
        polyglot=True,
        post_improve_diagnose=True,
    )
    assert metadata["infra_error_code"] == DISK_CODE
    assert metadata.get("model_patch_notempty") is not True
    saved = list(Path(tmp_path).glob("*/metadata.json"))
    assert saved
    blob = saved[0].read_text(encoding="utf-8")
    assert DISK_CODE in blob


def test_verify_gate_runs_before_the_polyglot_harness(monkeypatch, tmp_path):
    from self_improve_step import run_harness_polyglot

    def deny(*, phase):
        assert phase == "verify"
        raise ResourceGateError(DISK_CODE, "refusing verify: /tmp has 1 MiB free", None)

    monkeypatch.setattr("self_improve_step.require_resources_from_env", deny)

    def forbid_harness(*_args, **_kwargs):
        raise AssertionError("polyglot harness should not start")

    monkeypatch.setattr("self_improve_step.polyglot_harness", forbid_harness)
    with pytest.raises(ResourceGateError) as caught:
        run_harness_polyglot("go__poker", "child", [], 1, str(tmp_path), {}, "child", None, ["go__poker"], None)
    assert caught.value.code == DISK_CODE


def _live_args(tmp_path, extra=None):
    argv = [
        "--benchmark",
        "polyglot",
        "--shallow_eval",
        "--max_generation",
        "3",
        "--output_dir",
        str(tmp_path),
    ]
    if extra:
        argv.extend(extra)
    return parse_args(argv)


def _install_live_fakes(monkeypatch, tmp_path, improve):
    monkeypatch.setattr("EvoPilot_outer.load_task_lists", lambda _benchmark: (["go__poker"], ["go__poker"]))
    monkeypatch.setattr("EvoPilot_outer.prepare_initial_copy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "EvoPilot_outer.snapshot_from_initial_metadata",
        lambda *_args, **_kwargs: AgentSnapshot(
            agent_id="initial",
            parent_id=None,
            score=1.0,
            step=0,
            resolved_ids=("go__poker",),
            root=True,
        ),
    )
    monkeypatch.setattr("utils.container_runtime.resolve_container_runtime", lambda: "apptainer")
    monkeypatch.setattr("EvoPilot_outer.call_self_improve", improve)
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setenv("APPTAINER_WORKSPACE_ROOT", str(root))
    return root


def _regression_metadata(run_id="child"):
    return {
        "run_id": run_id,
        "parent_commit": "initial",
        "entry": "go__poker",
        "model_patch_notempty": True,
        "overall_performance": {
            "accuracy_score": 0.0,
            "total_submitted_instances": 1,
            "total_resolved_ids": [],
            "total_unresolved_ids": ["go__poker"],
            "total_emptypatch_ids": [],
        },
    }


def test_evopilot_low_tmp_does_not_call_self_improve(monkeypatch, tmp_path):
    calls = []

    def improve(**kwargs):
        calls.append(kwargs)
        return _regression_metadata()

    _install_live_fakes(monkeypatch, tmp_path, improve)

    def deny(thresholds=None, *, phase, probe=None):
        raise ResourceGateError(
            DISK_CODE,
            "refusing improve: /tmp has 1 MiB free. No model call and no new Apptainer sandbox will be started.",
            None,
        )

    monkeypatch.setattr("EvoPilot_outer.require_resources", deny)
    summary = run_live(_live_args(tmp_path), str(tmp_path))
    assert calls == []
    assert summary["infra_error_code"] == DISK_CODE
    assert "No model call" in summary["stop_reason"]
    stored = json.loads((tmp_path / "evopilot_summary.json").read_text(encoding="utf-8"))
    assert stored["infra_error_code"] == DISK_CODE
    assert summary["decisions"] == []


def test_evopilot_infra_verify_stops_before_the_next_improve(monkeypatch, tmp_path):
    calls = []
    root = None

    def improve(**kwargs):
        calls.append(kwargs["evopilot_role"])
        owned = root / f"dgm-container-{len(calls)}"
        owned.mkdir()
        stamp_workspace(owned, name=owned.name)
        foreign = root / "other-user"
        foreign.mkdir(exist_ok=True)
        (foreign / ".dgm-workspace-owner").write_text("owner=someone-else\n", encoding="utf-8")
        return apply_infra_metadata(
            {"run_id": f"child-{len(calls)}", "entry": "go__poker", "model_patch_notempty": False, "overall_performance": {}},
            INFRA_NOSPACE,
            "OSError: [Errno 28] No space left on device",
        )

    root = _install_live_fakes(monkeypatch, tmp_path, improve)
    monkeypatch.setattr("EvoPilot_outer.require_resources", lambda *args, **kwargs: None)
    summary = run_live(_live_args(tmp_path), str(tmp_path))
    assert calls == ["improve"]
    assert summary["infra_error_code"] == INFRA_NOSPACE
    assert summary["decisions"] == ["rollback"]
    assert not (root / "dgm-container-1").exists()
    assert (root / "other-user").is_dir()


def test_evopilot_continue_flag_skips_debug_but_tries_later_generations(monkeypatch, tmp_path):
    calls = []

    def improve(**kwargs):
        calls.append(kwargs["evopilot_role"])
        return apply_infra_metadata(
            {"run_id": f"child-{len(calls)}", "entry": "go__poker", "model_patch_notempty": False, "overall_performance": {}},
            INFRA_LDAP,
            "FATAL: failed to get user id: unknown userid 1000",
        )

    _install_live_fakes(monkeypatch, tmp_path, improve)
    monkeypatch.setattr("EvoPilot_outer.require_resources", lambda *args, **kwargs: None)
    summary = run_live(_live_args(tmp_path, ["--continue-on-infra-error"]), str(tmp_path))
    assert calls == ["improve", "improve", "improve"]
    assert summary["infra_error_code"] is None
    assert summary["decisions"] == ["rollback", "rollback", "rollback"]


def test_evopilot_bad_patch_still_rolls_back_and_continues(monkeypatch, tmp_path):
    calls = []
    root = None

    def improve(**kwargs):
        calls.append(kwargs["evopilot_role"])
        owned = root / "dgm-container-live"
        if owned.exists():
            owned.mkdir()
        else:
            owned.mkdir()
        stamp_workspace(owned, name="dgm-container-live")
        return _regression_metadata(f"child-{len(calls)}")

    root = _install_live_fakes(monkeypatch, tmp_path, improve)
    monkeypatch.setattr("EvoPilot_outer.require_resources", lambda *args, **kwargs: None)
    summary = run_live(_live_args(tmp_path), str(tmp_path))
    assert calls == ["improve", "improve", "improve"]
    assert summary["infra_error_code"] is None
    assert summary["stop_reason"] is None
    assert summary["decisions"] == ["rollback", "rollback", "rollback"]
    assert not (root / "dgm-container-live").exists()


def test_dry_resource_check_does_not_start_a_run(monkeypatch):
    def low():
        return _snapshot(free=1024)

    monkeypatch.setattr("utils.host_resources.snapshot_resources", low)

    def forbid(*_args, **_kwargs):
        raise AssertionError("live run should not start")

    monkeypatch.setattr("EvoPilot_outer.run_live", forbid)
    monkeypatch.setattr("EvoPilot_outer.run_smoke", forbid)
    code = main(["--dry-resource-check", "--min-tmp-free-gb", "10"])
    assert code == 2
    code = main(["--dry-resource-check", "--min-tmp-free-gb", "0", "--min-available-mem-gb", "0", "--max-mem-pressure", "-1"])
    assert code == 0


def _dgm_child(output_dir, code):
    folder = os.path.join(output_dir, "child")
    os.makedirs(folder, exist_ok=True)
    metadata = {
        "run_id": "child",
        "infra_error_code": code,
        "overall_performance": {},
    }
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    return metadata


def _patch_dgm(monkeypatch, tmp_path, improve, require):
    monkeypatch.setenv("DGM_OUTPUT_ROOT", str(tmp_path / "output_dgm"))

    def fake_initialize(output_dir, prevrun_dir=None, polyglot=False):
        os.makedirs(os.path.join(output_dir, "initial"), exist_ok=True)
        with open(os.path.join(output_dir, "initial", "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "overall_performance": {
                        "accuracy_score": 0.2,
                        "total_unresolved_ids": ["go__poker"],
                        "total_resolved_ids": [],
                        "total_emptypatch_ids": [],
                        "total_submitted_instances": 1,
                    }
                },
                handle,
            )
        return ["initial"], 0

    monkeypatch.setattr(DGM_outer, "initialize_run", fake_initialize)
    monkeypatch.setattr(DGM_outer, "require_resources", require)
    monkeypatch.setattr(DGM_outer, "cleanup_owned_workspaces", lambda *args, **kwargs: [])
    monkeypatch.setattr("self_improve_step.self_improve", improve)


def test_dgm_stops_after_infra_verify(monkeypatch, tmp_path):
    calls = []

    def improve(**kwargs):
        calls.append(kwargs)
        return _dgm_child(kwargs["output_dir"], INFRA_NOSPACE)

    _patch_dgm(monkeypatch, tmp_path, improve, lambda *args, **kwargs: None)
    code = DGM_outer.main(
        [
            "--polyglot",
            "--shallow_eval",
            "--no_full_eval",
            "--selfimprove_size",
            "1",
            "--selfimprove_workers",
            "1",
            "--max_generation",
            "4",
        ]
    )
    assert code == 2
    assert len(calls) == 1
    records = (tmp_path / "output_dgm").glob("*/dgm_metadata.jsonl")
    text = "".join(path.read_text(encoding="utf-8") for path in records)
    assert INFRA_NOSPACE in text
    assert "stop_reason" in text


def test_dgm_low_tmp_does_not_submit_self_improve(monkeypatch, tmp_path):
    calls = []

    def improve(**kwargs):
        calls.append(kwargs)
        return {"run_id": "should-not-run"}

    def deny(*_args, **_kwargs):
        raise ResourceGateError(DISK_CODE, "refusing improve: /tmp has 1 MiB free", None)

    _patch_dgm(monkeypatch, tmp_path, improve, deny)
    code = DGM_outer.main(["--polyglot", "--shallow_eval", "--no_full_eval", "--max_generation", "3"])
    assert code == 2
    assert calls == []
