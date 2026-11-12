"""Host resource gates, workspace ownership, and outer-loop fail-fast."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

import DGM_outer
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
    add_resource_arguments,
    apply_infra_metadata,
    assess_resources,
    classify_infra_error,
    cleanup_owned_workspaces,
    raise_for_infra_results,
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


def _thresholds(argv=()):
    parser = argparse.ArgumentParser()
    add_resource_arguments(parser)
    return thresholds_from_args(parser.parse_args(list(argv)))


def test_low_tmp_and_pressure_are_distinct_from_a_bad_patch():
    thresholds = _thresholds()
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

    disabled = _thresholds(["--min-tmp-free-gb", "0", "--min-available-mem-gb", "0", "--max-mem-pressure", "-1"])
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


def test_self_improve_records_an_unpack_that_gave_up(monkeypatch, tmp_path):
    """The first exec unpacks the sandbox; giving up there is a gate refusal, not a crash."""
    import self_improve_step
    from self_improve_step import self_improve

    workdir = tmp_path / "repo"
    (workdir / "swe_bench" / "subsets").mkdir(parents=True)
    (workdir / "swe_bench" / "subsets" / "big.json").write_text("[]", encoding="utf-8")
    (workdir / "polyglot").mkdir()
    (workdir / "polyglot" / "polyglot_benchmark_metadata.json").write_text("[]", encoding="utf-8")
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(self_improve_step, "dataset", None)

    class Container:
        def __init__(self):
            self.commands = []

        def start(self):
            pass

        def exec_run(self, command, workdir=None):
            self.commands.append(command)
            raise ResourceGateError(DISK_CODE, "refusing a sandbox unpack after waiting 3600s", None)

    container = Container()
    removed = []

    def forbid_diagnose(*_args, **_kwargs):
        raise AssertionError("diagnose should not call the model")

    monkeypatch.setattr("self_improve_step.require_resources_from_env", lambda *, phase: None)
    monkeypatch.setattr("self_improve_step.docker_from_env", lambda: object())
    monkeypatch.setattr("self_improve_step.remove_existing_container", lambda *_args: None)
    monkeypatch.setattr("self_improve_step.build_dgm_container", lambda *_args, **_kwargs: container)
    monkeypatch.setattr("self_improve_step.cleanup_container", removed.append)
    monkeypatch.setattr("self_improve_step.diagnose_problem", forbid_diagnose)

    metadata = self_improve(
        parent_commit="initial",
        output_dir=str(tmp_path / "out"),
        entry="go__poker",
        polyglot=True,
    )
    assert metadata["infra_error_code"] == DISK_CODE
    assert len(container.commands) == 1
    assert removed == [container]
    assert list((tmp_path / "out").glob("*/metadata.json"))


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


def test_dry_resource_check_does_not_start_a_run(monkeypatch):
    def low():
        return _snapshot(free=1024)

    monkeypatch.setattr("utils.host_resources.snapshot_resources", low)

    def forbid(*_args, **_kwargs):
        raise AssertionError("live run should not start")

    monkeypatch.setattr(DGM_outer, "initialize_run", forbid)
    code = DGM_outer.main(["--polyglot", "--dry-resource-check", "--min-tmp-free-gb", "10"])
    assert code == 2
    code = DGM_outer.main(
        ["--polyglot", "--dry-resource-check", "--min-tmp-free-gb", "0", "--min-available-mem-gb", "0", "--max-mem-pressure", "-1"]
    )
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

    def fake_initialize(output_dir, prevrun_dir=None, polyglot=False, initial_dir=None):
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
