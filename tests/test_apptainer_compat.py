"""Unit tests for the Apptainer docker-py shim. No daemon and no apptainer binary."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from utils.apptainer_compat import (
    ApptainerClient,
    ApptainerContainer,
    ExecResult,
    _apply_dockerfile_copy,
    _build_sif_from_dockerfile,
    _docker_uri_for_local_tag,
    _parse_dockerfile_instructions,
    _qualify_docker_image,
    _sif_path_for_tag,
)
from utils.container_runtime import clear_runtime_cache, resolve_container_runtime
from utils.docker_utils import docker_from_env


@pytest.fixture(autouse=True)
def _isolate_runtime(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    workspace = tmp_path / "workspaces"
    image_dir.mkdir()
    workspace.mkdir()
    monkeypatch.setenv("APPTAINER_IMAGE_DIR", str(image_dir))
    monkeypatch.setenv("APPTAINER_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("APPTAINER_TMPDIR", str(tmp_path))
    dummy = tmp_path / "bin" / "apptainer"
    dummy.parent.mkdir()
    dummy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dummy.chmod(0o755)
    monkeypatch.setenv("APPTAINER_BIN", str(dummy))
    monkeypatch.setenv("APPTAINER_BUILD_FAKEROOT", "0")
    monkeypatch.setenv("APPTAINER_CONTAIN", "1")
    monkeypatch.setenv("APPTAINER_NO_MOUNT", "tmp")
    monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
    clear_runtime_cache()
    yield
    clear_runtime_cache()


def _completed(args, code=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def test_explicit_and_auto_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "docker")
    assert resolve_container_runtime(refresh=True) == "docker"

    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "singularity")
    assert resolve_container_runtime(refresh=True) == "apptainer"

    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "auto")
    bin_dir = tmp_path / "path-bin"
    bin_dir.mkdir()
    apptainer = bin_dir / "apptainer"
    apptainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    apptainer.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("APPTAINER_BIN", raising=False)
    monkeypatch.setattr(
        "utils.container_runtime.docker_daemon_available", lambda: False
    )
    clear_runtime_cache()
    assert resolve_container_runtime(refresh=True) == "apptainer"

    monkeypatch.setattr(
        "utils.container_runtime.docker_daemon_available", lambda: True
    )
    assert resolve_container_runtime(refresh=True) == "docker"

    monkeypatch.setattr(
        "utils.container_runtime.docker_daemon_available", lambda: False
    )
    monkeypatch.setattr("utils.container_runtime.find_apptainer_bin", lambda: None)
    clear_runtime_cache()
    with pytest.raises(RuntimeError):
        resolve_container_runtime(refresh=True)


def test_docker_from_env_does_not_call_docker(monkeypatch):
    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "apptainer")

    def _boom(*_args, **_kwargs):
        raise AssertionError("docker.from_env should not be called")

    import docker

    monkeypatch.setattr(docker, "from_env", _boom)
    client = docker_from_env()
    assert isinstance(client, ApptainerClient)
    assert client.timeout == 600


def test_exception_aliases_match_docker():
    import docker.errors

    from utils import apptainer_errors

    assert apptainer_errors.ImageNotFound is docker.errors.ImageNotFound
    assert apptainer_errors.NotFound is docker.errors.NotFound
    assert apptainer_errors.APIError is docker.errors.APIError
    assert apptainer_errors.BuildError is docker.errors.BuildError


def test_sif_names_and_registry_uris():
    images = Path("/tmp/images")
    eval_name = _sif_path_for_tag(
        "sweb.eval.x86_64.django__django-10973:latest", images
    ).name
    assert eval_name == "sweb.eval.x86_64.django__django-10973_latest.sif"
    assert _sif_path_for_tag("dgm", images).name == "dgm_latest.sif"
    assert _docker_uri_for_local_tag(
        "sweb.eval.x86_64.django__django-10973:latest"
    ) == (
        "docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-10973:latest"
    )
    assert (
        _docker_uri_for_local_tag("buildpack-deps:jammy")
        == "docker://docker.io/library/buildpack-deps:jammy"
    )
    assert _qualify_docker_image("python:3.10-slim") == "docker.io/library/python:3.10-slim"


def test_dockerfile_parser_workdir_copy_and_continuations():
    text = """
# comment before FROM
FROM --platform=linux/amd64 pb.base.x86_64:latest
ENV PATH="/usr/local/go/bin:${PATH}"
RUN apt-get update && apt-get install -y \\
    git \\
    && true
WORKDIR /dgm
COPY . .
"""
    from_ref, copies, envs, runs = _parse_dockerfile_instructions(text)
    assert from_ref == "pb.base.x86_64:latest"
    assert copies == [(".", "/dgm")]
    assert envs == ['PATH="/usr/local/go/bin:${PATH}"']
    assert any("git" in line and "&& true" in line for line in runs)
    assert any("mkdir -p /dgm" in line for line in runs)


def test_copy_dot_does_not_replace_sandbox(tmp_path):
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "hello.txt").write_text("hi", encoding="utf-8")
    (context / "skip_me.txt").write_text("no", encoding="utf-8")
    (context / ".dockerignore").write_text("skip_me.txt\n", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    (sandbox / "bin").mkdir(parents=True)
    (sandbox / "bin" / "sh").write_text("#!/bin/sh\n", encoding="utf-8")

    _apply_dockerfile_copy(sandbox, context, ".", "/dgm")

    assert (sandbox / "bin" / "sh").is_file()
    assert (sandbox / "dgm" / "hello.txt").read_text(encoding="utf-8") == "hi"
    assert not (sandbox / "dgm" / "skip_me.txt").exists()


def test_build_sif_copies_into_workdir(tmp_path, monkeypatch):
    context = tmp_path / "repo"
    context.mkdir()
    (context / "hello.txt").write_text("payload", encoding="utf-8")
    (context / "Dockerfile").write_text(
        "FROM python:3.10-slim\nWORKDIR /dgm\nCOPY . .\nRUN echo built\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run(args, timeout=None, check=True):
        if "--sandbox" in args:
            sandbox = Path(args[args.index("--sandbox") + 1])
            (sandbox / "bin").mkdir(parents=True, exist_ok=True)
        if args[:1] == ["exec"] or (len(args) > 1 and args[0] == "exec"):
            sandbox = Path(args[args.index("bash") - 1])
            seen["exec_args"] = list(args)
            seen["contain"] = "--contain" in args
            seen["no_mount"] = args[args.index("--no-mount") + 1] == "tmp"
            seen["hello_during_exec"] = (sandbox / "dgm" / "hello.txt").is_file()
            seen["sandbox_root_intact"] = (sandbox / "bin").is_dir()
        return _completed(args, stdout=b"ok\n")

    real_popen = subprocess.Popen

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self._real = None
            if cmd and cmd[0] != "cp":
                sandbox = Path(cmd[-1])
                seen["hello_at_sif"] = (sandbox / "dgm" / "hello.txt").read_text(
                    encoding="utf-8"
                )
                Path(cmd[-2]).write_bytes(b"sif")
                self.stdout = io.BytesIO(b"built\n")
            else:
                self._real = real_popen(cmd, **kwargs)
                self.stdout = self._real.stdout

        def wait(self, timeout=None):
            if self._real is not None:
                return self._real.wait(timeout=timeout)
            return 0

        def __enter__(self):
            if self._real is not None:
                return self._real.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._real is not None:
                return self._real.__exit__(exc_type, exc, tb)
            return False

    monkeypatch.setattr("utils.apptainer_compat._run_apptainer", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    sif = tmp_path / "images" / "dgm_latest.sif"
    logs = _build_sif_from_dockerfile(context, "dgm", sif, timeout=5)
    assert seen["hello_during_exec"] is True
    assert seen["sandbox_root_intact"] is True
    assert seen["contain"] is True
    assert seen["no_mount"] is True
    assert "--fakeroot" not in seen["exec_args"]
    assert seen["hello_at_sif"] == "payload"
    assert sif.is_file()
    assert any("Copied . -> /dgm" in entry.get("stream", "") for entry in logs)


def _install_build_fakes(monkeypatch, seen):
    """Sandbox + post-script stand-ins. The final SIF pack writes an empty file."""

    def fake_run(args, timeout=None, check=True):
        if "--sandbox" in args:
            sandbox = Path(args[args.index("--sandbox") + 1])
            (sandbox / "bin").mkdir(parents=True, exist_ok=True)
            seen["sandbox_timeout"] = timeout
            return _completed(args, stdout=b"sandbox\n")
        if args and args[0] == "exec":
            seen["exec_args"] = list(args)
            seen["post_timeout"] = timeout
            stdout = seen.get("stdout", b"ok\n")
            stderr = seen.get("stderr", b"")
            code = seen.get("code", 0)
            return _completed(args, code=code, stdout=stdout, stderr=stderr)
        return _completed(args, stdout=b"ok\n")

    real_popen = subprocess.Popen

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self._real = None
            if cmd and cmd[0] != "cp":
                Path(cmd[-2]).write_bytes(b"sif")
                self.stdout = io.BytesIO(b"built\n")
            else:
                self._real = real_popen(cmd, **kwargs)
                self.stdout = self._real.stdout

        def wait(self, timeout=None):
            seen["sif_wait_timeout"] = timeout
            return self._real.wait(timeout=timeout) if self._real is not None else 0

        def __enter__(self):
            return self._real.__enter__() if self._real is not None else self

        def __exit__(self, exc_type, exc, tb):
            if self._real is not None:
                return self._real.__exit__(exc_type, exc, tb)
            return False

    monkeypatch.setattr("utils.apptainer_compat._run_apptainer", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)


def test_post_script_exec_includes_fakeroot_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("APPTAINER_BUILD_FAKEROOT", "1")
    context = tmp_path / "repo"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM buildpack-deps:jammy\nRUN echo hi\n",
        encoding="utf-8",
    )
    seen = {}
    _install_build_fakes(monkeypatch, seen)
    sif = tmp_path / "images" / "pb.base.x86_64_latest.sif"
    _build_sif_from_dockerfile(context, "pb.base.x86_64:latest", sif, timeout=5)
    args = seen["exec_args"]
    assert args[0] == "exec"
    assert "--fakeroot" in args
    assert args.index("--fakeroot") < args.index(str(Path(args[args.index("bash") - 1])))
    assert "--writable" in args
    assert "--contain" in args
    assert seen["post_timeout"] == 5
    assert seen["sif_wait_timeout"] == 5


def test_dockerfile_build_defaults_to_7200_seconds(tmp_path, monkeypatch):
    from utils.apptainer_compat import ApptainerClient, dockerfile_build_timeout

    monkeypatch.delenv("APPTAINER_BUILD_TIMEOUT", raising=False)
    monkeypatch.delenv("APPTAINER_API_TIMEOUT", raising=False)
    monkeypatch.delenv("CONTAINER_API_TIMEOUT", raising=False)
    assert dockerfile_build_timeout(600) == 7200

    monkeypatch.setenv("APPTAINER_BUILD_TIMEOUT", "9000")
    assert dockerfile_build_timeout(600) == 9000
    monkeypatch.delenv("APPTAINER_BUILD_TIMEOUT")
    monkeypatch.setenv("APPTAINER_API_TIMEOUT", "1800")
    assert dockerfile_build_timeout(1800) == 1800

    monkeypatch.delenv("APPTAINER_API_TIMEOUT")
    monkeypatch.delenv("CONTAINER_API_TIMEOUT", raising=False)
    monkeypatch.setenv("APPTAINER_BUILD_FAKEROOT", "0")
    context = tmp_path / "repo"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM buildpack-deps:jammy\nRUN echo hi\n",
        encoding="utf-8",
    )
    seen = {}
    _install_build_fakes(monkeypatch, seen)
    client = ApptainerClient(timeout=600)
    assert client.timeout == 600
    _image, _logs = client.images.build(path=context, tag="pb.base.x86_64:latest")
    assert seen["sandbox_timeout"] == 7200
    assert seen["post_timeout"] == 7200
    assert seen["sif_wait_timeout"] == 7200


def test_post_script_failure_includes_stdout_and_stderr(tmp_path, monkeypatch):
    from utils.apptainer_errors import BuildError
    from utils.apptainer_compat import ApptainerClient

    monkeypatch.setenv("APPTAINER_BUILD_FAKEROOT", "1")
    context = tmp_path / "repo"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM buildpack-deps:jammy\nRUN apt-get install -y git\n",
        encoding="utf-8",
    )
    seen = {
        "code": 100,
        "stdout": b"Reading package lists...\n",
        "stderr": b"E: Could not open lock file - are you root?\n",
    }
    _install_build_fakes(monkeypatch, seen)
    sif = tmp_path / "images" / "pb.base.x86_64_latest.sif"
    with pytest.raises(BuildError) as raised:
        _build_sif_from_dockerfile(context, "pb.base.x86_64:latest", sif, timeout=5)
    text = str(raised.value)
    assert "Could not open lock file" in text
    assert "Reading package lists" in text
    assert "Could not open lock file" in raised.value.build_log
    assert "--fakeroot" in seen["exec_args"]

    client = ApptainerClient(timeout=5)

    def _boom(*_args, **_kwargs):
        raise raised.value

    monkeypatch.setattr(client.images, "build", _boom)
    chunks = list(client.api.build(path=str(context), tag="pb.base.x86_64:latest"))
    streamed = "".join(chunk.get("stream", "") for chunk in chunks)
    assert "Could not open lock file" in streamed
    assert "Reading package lists" in streamed
    assert chunks[-1]["errorDetail"]["message"]
    assert "Could not open lock file" in chunks[-1]["errorDetail"]["message"]


def test_pull_is_locked_and_renames_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "apptainer")
    calls = []

    def fake_run(args, timeout=None, check=True):
        calls.append(list(args))
        if args[0] == "pull":
            Path(args[1]).write_bytes(b"sif-bytes")
        return _completed(args, stdout=b"pulled\n")

    monkeypatch.setattr("utils.apptainer_compat._run_apptainer", fake_run)
    client = ApptainerClient(timeout=10)
    first = client.images._pull_to_sif("alpine:latest")
    second = client.images._pull_to_sif("alpine:latest")
    assert first == second
    assert first.name == "alpine_latest.sif"
    assert first.read_bytes() == b"sif-bytes"
    assert not first.with_suffix(".sif.tmp").exists()
    assert [c for c in calls if c[0] == "pull"] == [
        ["pull", str(first.with_suffix(".sif.tmp")), "docker://docker.io/library/alpine:latest"]
    ]
    meta = json.loads(first.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["tags"] == ["alpine:latest"]
    image = client.images.get("alpine:latest")
    assert "alpine:latest" in image.tags
    assert image.history() == []
    assert isinstance(image.attrs["Created"], float)


def test_missing_env_image_is_not_invented():
    client = ApptainerClient(timeout=5)
    from utils.apptainer_errors import ImageNotFound

    with pytest.raises(ImageNotFound):
        client.images.get("sweb.env.x86_64.missing:latest")


def test_remove_keeps_sweb_eval_images(tmp_path):
    client = ApptainerClient(timeout=5)
    image_dir = Path(os.environ["APPTAINER_IMAGE_DIR"])
    eval_sif = image_dir / "sweb.eval.x86_64.django__django-10973_latest.sif"
    eval_sif.write_bytes(b"eval")
    client.images._write_meta(eval_sif, ["sweb.eval.x86_64.django__django-10973:latest"])
    client.images.remove("sweb.eval.x86_64.django__django-10973:latest")
    assert eval_sif.exists()

    pb = image_dir / "pb.eval.x86_64.exercise_latest.sif"
    pb.write_bytes(b"pb")
    client.images._write_meta(pb, ["pb.eval.x86_64.exercise:latest"])
    client.images.remove("pb.eval.x86_64.exercise:latest")
    assert not pb.exists()


def test_exec_argv_contain_and_user(monkeypatch):
    client = ApptainerClient(timeout=5)
    container = client.containers.create(
        image="alpine:latest",
        name="dgm-container-smoke",
        command="tail -f /dev/null",
        user="root",
        detach=True,
    )
    (container._workspace / "rootfs" / "bin").mkdir(parents=True)
    calls = []

    def fake_run(cmd, capture_output=True, timeout=None, **kwargs):
        calls.append(cmd)
        return _completed(cmd, stdout=b"ok\n")

    monkeypatch.setattr("utils.apptainer_compat.subprocess.run", fake_run)
    result = container.exec_run("echo ok", user="root")
    assert result.exit_code == 0
    cmd = calls[0]
    assert "exec" in cmd
    assert "--writable" in cmd
    assert "--contain" in cmd
    assert cmd[cmd.index("--no-mount") + 1] == "tmp"
    assert "echo ok" in cmd
    assert "su" not in cmd

    container.exec_run("id", user="nonroot")
    wrapped = " ".join(calls[1])
    assert "su" in wrapped and "nonroot" in wrapped


def test_put_and_get_archive_roundtrip():
    client = ApptainerClient(timeout=5)
    container = client.containers.create(image="alpine:latest", name="copy-test")
    (container._workspace / "rootfs" / "bin").mkdir(parents=True)
    payload = b"patch-body"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="patch.diff")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    assert container.put_archive("/tmp", buf.getvalue()) is True
    written = container._workspace / "rootfs" / "tmp" / "patch.diff"
    assert written.read_bytes() == payload
    assert (container._workspace / "rootfs" / "bin").is_dir()

    chunks, stat = container.get_archive("/tmp/patch.diff")
    assert stat["name"] == "patch.diff"
    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks)), mode="r") as tar:
        member = tar.extractfile(tar.getmembers()[0])
        assert member is not None
        assert member.read() == payload


def test_exec_run_with_timeout_uses_timeout_binary():
    from polyglot.docker_utils import exec_run_with_timeout

    client = ApptainerClient(timeout=5)
    container = client.containers.create(image="alpine:latest", name="timeout-test")
    seen = {}

    def fake_exec(cmd, workdir=None, environment=None, detach=False, **kwargs):
        seen["cmd"] = cmd
        seen["workdir"] = workdir
        return ExecResult(exit_code=124, output=b"killed")

    container.exec_run = fake_exec  # type: ignore
    text, timed_out, _elapsed = exec_run_with_timeout(container, ["sleep", "100"], timeout=5)
    assert timed_out is True
    assert text == "killed"
    assert seen["cmd"].startswith("timeout 5 ")
    assert "sleep" in seen["cmd"]
    assert seen["workdir"] == "/"


def test_swebench_eval_argv_switches_with_runtime(monkeypatch):
    from swe_bench.report import swebench_eval_argv

    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "apptainer")
    argv = swebench_eval_argv("/repo", "preds.jsonl", "princeton-nlp/SWE-bench_Verified", 2, "run1")
    assert argv[1:3] == ["-m", "swe_bench.run_evaluation_apptainer"]
    assert "--namespace" not in argv

    monkeypatch.setenv("DGM_CONTAINER_RUNTIME", "docker")
    argv = swebench_eval_argv("/repo", "preds.jsonl", "ds", 2, "run1")
    assert argv[1].endswith("swe_bench/SWE-bench/swebench/harness/run_evaluation.py")


def test_build_skips_existing_sif_unless_nocache(tmp_path, monkeypatch):
    client = ApptainerClient(timeout=5)
    image_dir = Path(os.environ["APPTAINER_IMAGE_DIR"])
    sif = _sif_path_for_tag("dgm", image_dir)
    sif.write_bytes(b"old")
    client.images._write_meta(sif, ["dgm"])
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    def _fail(*_args, **_kwargs):
        raise AssertionError("rebuild should be skipped")

    monkeypatch.setattr("utils.apptainer_compat._build_sif_from_dockerfile", _fail)
    image, logs = client.images.build(path=context, tag="dgm", nocache=False)
    assert logs == []
    assert "dgm" in image.tags
    assert sif.read_bytes() == b"old"


def test_recreate_container_clears_workspace():
    client = ApptainerClient(timeout=5)
    first = client.containers.create(image="alpine:latest", name="again")
    (first._workspace / "dirty").write_text("x", encoding="utf-8")
    second = client.containers.create(image="alpine:latest", name="again")
    assert not (second._workspace / "dirty").exists()
    second.remove()
    with pytest.raises(Exception):
        client.containers.get("again")
    assert "again" not in client.containers._registry
