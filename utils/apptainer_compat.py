"""Docker-py compatible client implemented with Apptainer.

Call sites keep using ``images.build``, ``containers.create``, ``exec_run``,
``put_archive``, and ``get_archive``. Under the hood this pulls OCI images to
``.sif`` files, builds a writable sandbox, and runs ``apptainer exec``.

Persistent containers are per-name workspace directories plus a sandbox rootfs,
not long-lived ``apptainer instance`` processes.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from utils import apptainer_errors as container_errors
from utils.fs_copy import copytree_for_build


GHCR_EPOCH_PREFIX = os.environ.get(
    "SWE_GHCR_EPOCH_PREFIX", "ghcr.io/epoch-research"
)

_PULL_LOCKS: Dict[str, threading.Lock] = {}
_PULL_LOCKS_GUARD = threading.Lock()


def get_apptainer_bin() -> str:
    return os.environ.get("APPTAINER_BIN") or shutil.which("apptainer") or shutil.which("singularity") or "apptainer"


def get_workspace_root() -> Path:
    return Path(
        os.environ.get(
            "APPTAINER_WORKSPACE_ROOT",
            str(Path(os.environ.get("TMPDIR", "/tmp")) / "apptainer-workspaces"),
        )
    )


def _use_fakeroot_build() -> bool:
    return os.environ.get("APPTAINER_BUILD_FAKEROOT", "1") == "1"


def _use_nv() -> bool:
    return os.environ.get("APPTAINER_NV", "0") == "1"


def _use_host_network() -> bool:
    return os.environ.get("APPTAINER_USE_HOST_NETWORK", "0") == "1"


def _contain_enabled() -> bool:
    """Keep image ``/tmp`` and ``$HOME`` inside the writable sandbox.

    Apptainer bind-mounts the host ``/tmp`` and ``$HOME`` by default, which
    hides files the harness writes at those paths (Polyglot's ``/tmp/patch.diff``).
    """
    return os.environ.get("APPTAINER_CONTAIN", "1") == "1"


def _no_mount_spec() -> str:
    raw = os.environ.get("APPTAINER_NO_MOUNT", "tmp").strip()
    if raw.lower() in ("", "0", "none", "off"):
        return ""
    return raw


def get_image_dir() -> Path:
    """Resolve Apptainer SIF storage. Re-read on each call so tests can retarget it."""
    if os.environ.get("APPTAINER_IMAGE_DIR"):
        return Path(os.environ["APPTAINER_IMAGE_DIR"])
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home())))
    candidates = [
        hf_home / "apptainer_images",
        hf_home / ".cache" / "huggingface" / "apptainer_images",
        Path.home() / ".cache" / "huggingface" / "apptainer_images",
        Path.home() / ".apptainer" / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.sif")):
            return candidate
    return hf_home / "apptainer_images"


def _candidate_image_dirs() -> List[Path]:
    dirs: List[Path] = []
    seen: set[str] = set()
    explicit = os.environ.get("APPTAINER_IMAGE_DIR", "")
    for candidate in [
        get_image_dir(),
        Path(explicit) if explicit else None,
        Path.home() / ".cache" / "huggingface" / "apptainer_images",
        Path.home() / ".apptainer" / "images",
    ]:
        if candidate is None or not str(candidate) or str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _lock_for_path(path: Path) -> threading.Lock:
    key = str(path)
    with _PULL_LOCKS_GUARD:
        if key not in _PULL_LOCKS:
            _PULL_LOCKS[key] = threading.Lock()
        return _PULL_LOCKS[key]


def _run_apptainer(
    args: List[str],
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [get_apptainer_bin()] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        timeout=timeout,
        check=check,
    )


def _sanitize_tag(image_ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", image_ref)


def _split_image_ref(image_ref: str) -> Tuple[str, str]:
    if ":" in image_ref and not image_ref.startswith("docker://"):
        repo, tag = image_ref.rsplit(":", 1)
        return repo, tag
    return image_ref, "latest"


def _sif_path_for_tag(image_ref: str, image_dir: Optional[Path] = None) -> Path:
    repo, tag = _split_image_ref(image_ref)
    safe = _sanitize_tag(f"{repo}_{tag}")
    return (image_dir or get_image_dir()) / f"{safe}.sif"


def _qualify_docker_image(image_ref: str) -> str:
    image_ref = image_ref.strip()
    if image_ref.startswith("docker://"):
        image_ref = image_ref.split("://", 1)[1]
    repo, tag = _split_image_ref(image_ref)
    image_ref = f"{repo}:{tag}"
    if "/" not in repo:
        return f"docker.io/library/{image_ref}"
    registry = repo.split("/", 1)[0]
    if "." not in registry and ":" not in registry:
        return f"docker.io/{image_ref}"
    return image_ref


def _update_workdir(current: str, arg: str) -> str:
    arg = arg.strip().strip('"').strip("'")
    if arg.startswith("/"):
        new = arg
    else:
        base = current.rstrip("/") or "/"
        new = f"{base}/{arg}" if base != "/" else f"/{arg}"
    return new.rstrip("/") or "/"


def _parse_dockerfile_instructions(
    dockerfile_text: str,
) -> Tuple[Optional[str], List[tuple[str, str]], List[str], List[str]]:
    """Parse a Dockerfile into FROM, COPY steps, ENV lines, and RUN commands.

    COPY destinations are resolved against the WORKDIR in effect at that line,
    so ``WORKDIR /dgm`` followed by ``COPY . .`` lands in ``/dgm``.
    """
    from_ref: Optional[str] = None
    copy_steps: List[tuple[str, str]] = []
    env_lines: List[str] = []
    run_lines: List[str] = []
    workdir = "/"

    physical_lines = dockerfile_text.splitlines()
    idx = 0
    while idx < len(physical_lines):
        raw = physical_lines[idx]
        idx += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split(None, 1)
        instr = parts[0].upper()
        payload = parts[1] if len(parts) > 1 else ""
        while payload.endswith("\\") and idx < len(physical_lines):
            next_line = physical_lines[idx].strip()
            idx += 1
            if next_line.startswith("#"):
                continue
            payload = payload[:-1].rstrip() + " " + next_line

        if instr == "FROM" and from_ref is None:
            from_ref = payload.strip()
            if "--platform=" in from_ref:
                from_ref = from_ref.split(None, 1)[-1]
            continue
        if instr == "COPY":
            copy_parts = shlex.split(payload)
            if len(copy_parts) >= 2:
                src, dest = copy_parts[0], copy_parts[-1]
                if not dest.startswith("/"):
                    if dest in (".", "./"):
                        dest = workdir
                    else:
                        base = workdir.rstrip("/") or "/"
                        dest = f"{base}/{dest}" if base != "/" else f"/{dest}"
                copy_steps.append((src, dest))
            continue
        if instr == "ENV":
            env_lines.append(payload.strip())
            continue
        if instr == "ADD":
            add_parts = payload.split(None, 1)
            if len(add_parts) == 2 and add_parts[0].startswith("http"):
                run_lines.append(f'curl -fsSL "{add_parts[0]}" -o {add_parts[1]}')
            continue
        if instr == "RUN":
            run_lines.append(payload.strip())
            continue
        if instr == "WORKDIR":
            workdir = _update_workdir(workdir, payload)
            run_lines.append(f"mkdir -p {shlex.quote(workdir)} && cd {shlex.quote(workdir)}")

    return from_ref, copy_steps, env_lines, run_lines


def _dockerignore_patterns(context: Path) -> List[str]:
    path = context / ".dockerignore"
    if not path.is_file():
        return []
    patterns: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _path_ignored(rel: str, patterns: List[str]) -> bool:
    """Best-effort ``.dockerignore`` match for build-context copies."""
    import fnmatch

    rel = rel.replace("\\", "/").lstrip("./")
    name = rel.rsplit("/", 1)[-1]
    ignored = False
    for pat in patterns:
        negate = pat.startswith("!")
        body = pat[1:] if negate else pat
        directory_only = body.endswith("/")
        body = body.rstrip("/")
        glob_body = body.replace("**/", "*").replace("**", "*")
        matched = (
            rel == body
            or rel.startswith(body + "/")
            or fnmatch.fnmatch(rel, glob_body)
            or fnmatch.fnmatch(name, glob_body)
            or fnmatch.fnmatch(rel, glob_body.rstrip("/") + "/*")
        )
        if directory_only and matched and "." in name and "/" not in rel:
            matched = False
        if matched:
            ignored = not negate
    return ignored


def _copy_context_tree(src: Path, dst: Path, context: Path, patterns: List[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        rel = str(child.relative_to(context))
        if patterns and _path_ignored(rel, patterns):
            continue
        target = dst / child.name
        if child.is_dir() and not child.is_symlink():
            if patterns:
                _copy_context_tree(child, target, context, patterns)
            else:
                copytree_for_build(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", str(child), str(target)], check=True)


def _apply_dockerfile_copy(sandbox: Path, build_context: Path, src: str, dest: str) -> None:
    """Apply one Dockerfile COPY into the sandbox.

    ``COPY . .`` after ``WORKDIR /dgm`` copies build-context children into
    ``sandbox/dgm``. It must not replace the sandbox root (``cp -a`` / ``rmtree``
    of ``.`` would delete the base image rootfs).
    """
    src_path = build_context / src
    if not src_path.exists():
        raise container_errors.BuildError(
            f"COPY source not found: {src_path}",
            src,
        )
    dest_rel = dest.lstrip("/")
    target = sandbox if dest_rel in ("", ".") else sandbox / dest_rel
    patterns = _dockerignore_patterns(build_context)
    copy_children = src in (".", "./") or dest.endswith("/.")

    if copy_children and src_path.is_dir():
        if target.resolve() == sandbox.resolve():
            _copy_context_tree(src_path, sandbox, build_context, patterns)
            return
        if target.exists():
            shutil.rmtree(target)
        if patterns:
            _copy_context_tree(src_path, target, build_context, patterns)
        else:
            copytree_for_build(src_path, target)
        return

    if dest.endswith("/") or (target.exists() and target.is_dir()):
        target.mkdir(parents=True, exist_ok=True)
        target = target / src_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        if patterns:
            _copy_context_tree(src_path, target, build_context, patterns)
        else:
            copytree_for_build(src_path, target)
    else:
        shutil.copy2(src_path, target)


def _dockerfile_env_script(env_lines: List[str]) -> str:
    lines = ["#!/bin/sh"]
    for env in env_lines:
        if "=" not in env:
            continue
        key, val = env.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        escaped = val.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        lines.append(f'export {key}="{escaped}"')
    lines.append("")
    return "\n".join(lines)


def _exec_prefix() -> List[str]:
    args = ["exec", "--writable"]
    if _contain_enabled():
        args.append("--contain")
    no_mount = _no_mount_spec()
    if no_mount:
        args.extend(["--no-mount", no_mount])
    return args


def _process_output(proc: subprocess.CompletedProcess) -> str:
    """Combine stdout and stderr so a failed build log shows the real error."""

    def _decode(buf: Any) -> str:
        if not buf:
            return ""
        if isinstance(buf, str):
            return buf
        return bytes(buf).decode("utf-8", errors="replace")

    out = _decode(proc.stdout)
    err = _decode(proc.stderr)
    if out and err and err.strip() and err.strip() not in out:
        if not out.endswith("\n"):
            out += "\n"
        return out + err
    return out or err


def _raise_build_failure(message: str, output: str, logs: List[dict]) -> None:
    text = output or ""
    if text and not text.endswith("\n"):
        text += "\n"
    if text:
        logs.append({"stream": text})
    detail = f"{message}\n{text}" if text else message
    build_log = "".join(entry.get("stream", "") for entry in logs)
    raise container_errors.BuildError(detail, build_log)


def _build_error_stream(exc: BaseException) -> Iterator[dict]:
    """Turn a BuildError into docker-py build chunks, including captured output."""
    build_log = getattr(exc, "build_log", "") or ""
    if isinstance(build_log, bytes):
        build_log = build_log.decode("utf-8", errors="replace")
    elif isinstance(build_log, list):
        parts: List[str] = []
        for item in build_log:
            if isinstance(item, dict):
                parts.append(str(item.get("stream") or item.get("error") or ""))
            else:
                parts.append(str(item))
        build_log = "".join(parts)
    else:
        build_log = str(build_log)
    message = str(getattr(exc, "msg", None) or exc)
    if build_log:
        yield {"stream": build_log if build_log.endswith("\n") else build_log + "\n"}
    elif message:
        yield {"stream": message if message.endswith("\n") else message + "\n"}
    yield {"errorDetail": {"message": message}}


def _build_sif_from_dockerfile(
    path: Path,
    tag: str,
    sif: Path,
    timeout: Optional[int] = None,
) -> List[dict]:
    dockerfile_text = (path / "Dockerfile").read_text(encoding="utf-8")
    from_ref, copy_steps, env_lines, run_lines = _parse_dockerfile_instructions(
        dockerfile_text
    )
    if not from_ref:
        raise container_errors.BuildError(
            f"No FROM line in Dockerfile under {path}", "missing FROM"
        )

    logs: List[dict] = []
    sandbox = Path(
        tempfile.mkdtemp(
            prefix="apptainer-dockerfile-",
            dir=os.environ.get("APPTAINER_TMPDIR", os.environ.get("TMPDIR", "/tmp")),
        )
    )
    try:
        local_parent = _sif_path_for_tag(from_ref)
        if not local_parent.exists():
            for image_dir in _candidate_image_dirs():
                candidate = _sif_path_for_tag(from_ref, image_dir)
                if candidate.exists():
                    local_parent = candidate
                    break
        if local_parent.exists():
            logs.append({"stream": f"Building sandbox from local image {from_ref}\n"})
            proc = _run_apptainer(
                ["build", "--force", "--sandbox", str(sandbox), str(local_parent)],
                timeout=timeout,
                check=False,
            )
        else:
            base_uri = f"docker://{_qualify_docker_image(from_ref)}"
            logs.append({"stream": f"Building sandbox from {base_uri}\n"})
            proc = _run_apptainer(
                ["build", "--force", "--sandbox", str(sandbox), base_uri],
                timeout=timeout,
                check=False,
            )
        sandbox_output = _process_output(proc)
        if proc.returncode != 0:
            _raise_build_failure(
                f"apptainer sandbox build failed for {tag}",
                sandbox_output,
                logs,
            )
        elif sandbox_output:
            logs.append(
                {
                    "stream": sandbox_output
                    if sandbox_output.endswith("\n")
                    else sandbox_output + "\n"
                }
            )

        for src, dest in copy_steps:
            _apply_dockerfile_copy(sandbox, path, src, dest)
            logs.append({"stream": f"Copied {src} -> {dest}\n"})

        post_lines = [
            "#!/bin/bash",
            "set -euxo pipefail",
            "export HOME=/root",
        ]
        for env in env_lines:
            if "=" in env:
                key, val = env.split("=", 1)
                post_lines.append(
                    f'export {key.strip()}="{val.strip().strip(chr(34)).strip(chr(39))}"'
                )
        post_lines.extend(run_lines)
        post_lines.append("")
        post_path = sandbox / "dockerfile-post.sh"
        post_path.write_text("\n".join(post_lines), encoding="utf-8")
        if run_lines or env_lines:
            logs.append({"stream": "Running Dockerfile post script in sandbox\n"})
            # RUN steps (apt, conda, rustup, pip) install as root. --fakeroot maps
            # the caller to uid 0 inside the sandbox. On hosts without /etc/subuid
            # this still works via Apptainer's root-mapped user namespace.
            exec_args = _exec_prefix()
            if _use_fakeroot_build():
                exec_args = exec_args[:1] + ["--fakeroot"] + exec_args[1:]
            exec_args = exec_args + [str(sandbox), "bash", "/dockerfile-post.sh"]
            proc = _run_apptainer(exec_args, timeout=timeout, check=False)
            post_output = _process_output(proc)
            if proc.returncode != 0:
                _raise_build_failure(
                    f"apptainer Dockerfile post script failed for {tag}",
                    post_output,
                    logs,
                )
            elif post_output:
                logs.append(
                    {
                        "stream": post_output
                        if post_output.endswith("\n")
                        else post_output + "\n"
                    }
                )

        if env_lines:
            env_dir = sandbox / ".singularity.d" / "env"
            env_dir.mkdir(parents=True, exist_ok=True)
            env_path = env_dir / "91-dockerfile-env.sh"
            env_path.write_text(_dockerfile_env_script(env_lines), encoding="utf-8")
            logs.append({"stream": f"Persisted Dockerfile ENV to {env_path}\n"})

        build_args = ["build", "--force"]
        if _use_fakeroot_build():
            build_args.append("--fakeroot")
        build_args.extend([str(sif), str(sandbox)])
        proc = subprocess.Popen(
            [get_apptainer_bin()] + build_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace")
            logs.append({"stream": text})
        if proc.wait() != 0:
            raise container_errors.BuildError(
                f"apptainer build failed for {tag}",
                "".join(entry.get("stream", "") for entry in logs),
            )
        return logs
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _docker_uri_for_local_tag(image_ref: str) -> str:
    """Map local SWE-bench tags to registry URIs for ``apptainer pull``."""
    repo, tag = _split_image_ref(image_ref)
    if repo.startswith("ghcr.io/") or repo.startswith("docker.io/"):
        return f"docker://{repo}:{tag}"
    if "/" in repo and not repo.startswith("sweb.") and not repo.startswith("swebench"):
        return f"docker://{repo}:{tag}"
    if repo.startswith(("sweb.eval.", "sweb.base.", "sweb.env.")):
        suffix = repo[len("sweb.") :]
        remote = f"{GHCR_EPOCH_PREFIX}/swe-bench.{suffix}:{tag}"
        return f"docker://{remote}"
    if "/" not in repo:
        return f"docker://docker.io/library/{repo}:{tag}"
    return f"docker://{repo}:{tag}"


@dataclass
class ExecResult:
    exit_code: int = 0
    output: Union[bytes, Iterator[bytes]] = b""

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        if isinstance(self.output, bytes):
            return self.output.decode(encoding, errors)
        return b"".join(self.output).decode(encoding, errors)


@dataclass
class ApptainerImage:
    tags: List[str] = field(default_factory=list)
    id: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.id[:12] if self.id else ""

    def history(self) -> List[dict]:
        """SIF images have no docker layer history. Dependents are not inferred."""
        return []


def _read_meta(sif: Path) -> dict:
    meta_path = sif.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ApptainerImagesAPI:
    def __init__(self, client: "ApptainerClient") -> None:
        self._client = client

    def get(self, name: str) -> ApptainerImage:
        sif = self._client.images.resolve_sif(name)
        meta = _read_meta(sif)
        created = meta.get("created", 0) or 0
        tags = list(meta.get("tags") or [name])
        if name not in tags:
            tags.append(name)
        return ApptainerImage(
            tags=tags,
            id=sif.stem,
            attrs={"Created": created, "Id": sif.stem},
        )

    def pull(self, repository: str, platform: Optional[str] = None) -> ApptainerImage:
        sif = self._client.images._pull_to_sif(repository, platform=platform)
        return ApptainerImage(tags=[repository], id=sif.stem, attrs={"Created": time.time()})

    def remove(self, image_id: str, force: bool = False) -> None:
        preserve_eval = os.environ.get("APPTAINER_REMOVE_EVAL_IMAGES", "0") != "1"
        if preserve_eval and "sweb.eval." in str(image_id):
            return
        sif = _sif_path_for_tag(image_id)
        if not sif.exists():
            try:
                sif = self._client.images.resolve_sif(image_id)
            except container_errors.ImageNotFound:
                matched = None
                for image_dir in _candidate_image_dirs():
                    candidate = image_dir / f"{_sanitize_tag(image_id)}.sif"
                    if candidate.exists():
                        matched = candidate
                        break
                if matched is None:
                    raise container_errors.ImageNotFound(
                        f"Apptainer image not found: {image_id}"
                    )
                sif = matched
        if sif.exists():
            sif.unlink()
        meta = sif.with_suffix(".json")
        if meta.exists():
            meta.unlink()

    def list(self, all: bool = True) -> List[ApptainerImage]:
        images: List[ApptainerImage] = []
        seen: set[str] = set()
        for image_dir in _candidate_image_dirs():
            for sif in image_dir.glob("*.sif"):
                if str(sif) in seen:
                    continue
                seen.add(str(sif))
                meta = _read_meta(sif)
                images.append(
                    ApptainerImage(
                        tags=list(meta.get("tags") or []),
                        id=sif.stem,
                        attrs={"Created": meta.get("created", 0) or 0, "Id": sif.stem},
                    )
                )
        return images


class ApptainerImages:
    def __init__(self, client: "ApptainerClient") -> None:
        self._client = client

    def get(self, name: str) -> ApptainerImage:
        return ApptainerImagesAPI(self._client).get(name)

    def pull(self, repository: str, platform: Optional[str] = None) -> ApptainerImage:
        return ApptainerImagesAPI(self._client).pull(repository, platform=platform)

    def remove(self, image_id: str, force: bool = False) -> None:
        ApptainerImagesAPI(self._client).remove(image_id, force=force)

    def list(self, all: bool = True) -> List[ApptainerImage]:
        return ApptainerImagesAPI(self._client).list(all=all)

    def resolve_sif(self, image_ref: str) -> Path:
        for image_dir in _candidate_image_dirs():
            sif = _sif_path_for_tag(image_ref, image_dir)
            if sif.exists():
                return sif
        alias = self._resolve_multilingual_sif(image_ref)
        if alias is not None:
            return alias
        alias = self._resolve_sif_by_instance_suffix(image_ref)
        if alias is not None:
            return alias
        raise container_errors.ImageNotFound(f"Apptainer image not found: {image_ref}")

    def _resolve_multilingual_sif(self, image_ref: str) -> Optional[Path]:
        """Map SWE-bench Multilingual keys onto ``org_1776_rest`` SIF names."""
        if not image_ref.startswith("sweb.eval.x86_64."):
            return None
        instance_part = image_ref.split("sweb.eval.x86_64.", 1)[-1].split(":", 1)[0]
        if "__" not in instance_part:
            return None
        org, rest = instance_part.split("__", 1)
        ml_suffix = f"{org}_1776_{rest}"
        filename_patterns = [
            f"swebench_sweb.eval.x86_64.{ml_suffix}_latest.sif",
            f"sweb.eval.x86_64.{ml_suffix}_latest.sif",
        ]
        for image_dir in _candidate_image_dirs():
            for pattern in filename_patterns:
                candidate = image_dir / pattern
                if candidate.exists():
                    return candidate
            for meta_path in image_dir.glob("*.json"):
                tags = _read_meta(meta_path.with_suffix(".sif")).get("tags", [])
                if not any(ml_suffix in tag for tag in tags):
                    continue
                candidate = meta_path.with_suffix(".sif")
                if candidate.exists():
                    return candidate
        return None

    def _resolve_sif_by_instance_suffix(self, image_ref: str) -> Optional[Path]:
        match = re.search(r"([\w]+-\d+)(?::latest)?$", image_ref.split("/")[-1])
        if not match:
            return None
        suffix = match.group(1)
        for image_dir in _candidate_image_dirs():
            for meta_path in image_dir.glob("*.json"):
                tags = _read_meta(meta_path.with_suffix(".sif")).get("tags", [])
                if not any(suffix in str(tag) for tag in tags):
                    continue
                candidate = meta_path.with_suffix(".sif")
                if candidate.exists():
                    return candidate
        return None

    def ensure(
        self,
        name: str,
        pull: bool = True,
        platform: Optional[str] = None,
    ) -> Path:
        try:
            return self.resolve_sif(name)
        except container_errors.ImageNotFound:
            if not pull:
                raise
        self.pull(name, platform=platform)
        return self.resolve_sif(name)

    def _pull_to_sif(self, image_ref: str, platform: Optional[str] = None) -> Path:
        image_dir = get_image_dir()
        image_dir.mkdir(parents=True, exist_ok=True)
        sif = _sif_path_for_tag(image_ref, image_dir)
        if sif.exists():
            return sif
        with _lock_for_path(sif):
            if sif.exists():
                return sif
            tmp = sif.with_suffix(".sif.tmp")
            if tmp.exists():
                tmp.unlink()
            uri = _docker_uri_for_local_tag(image_ref)
            pull_args = ["pull", str(tmp), uri]
            if platform:
                pull_args = ["pull", "--platform", platform, str(tmp), uri]
            proc = _run_apptainer(pull_args, timeout=self._client.timeout, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
                if tmp.exists():
                    tmp.unlink()
                raise container_errors.APIError(f"apptainer pull failed: {err}")
            tmp.rename(sif)
            self._write_meta(sif, [image_ref])
        return sif

    def build(
        self,
        path: Union[str, Path] = ".",
        tag: Optional[str] = None,
        rm: bool = True,
        nocache: bool = False,
        **kwargs: Any,
    ) -> Tuple[ApptainerImage, List[dict]]:
        path = Path(path).resolve()
        if not tag:
            tag = path.name
        image_dir = get_image_dir()
        image_dir.mkdir(parents=True, exist_ok=True)
        sif = _sif_path_for_tag(tag, image_dir)
        if sif.exists() and not nocache:
            meta = _read_meta(sif)
            return (
                ApptainerImage(
                    tags=list(meta.get("tags") or [tag]),
                    id=sif.stem,
                    attrs={"Created": meta.get("created", 0) or 0, "Id": sif.stem},
                ),
                [],
            )
        if sif.exists() and nocache:
            sif.unlink()
            meta_path = sif.with_suffix(".json")
            if meta_path.exists():
                meta_path.unlink()

        dockerfile = path / "Dockerfile"
        if not dockerfile.exists():
            raise container_errors.BuildError(
                f"No Dockerfile in {path}", "missing Dockerfile"
            )

        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        from_ref, _, _, _ = _parse_dockerfile_instructions(dockerfile_text)
        if from_ref:
            logs = _build_sif_from_dockerfile(
                path, tag, sif, timeout=self._client.timeout
            )
        else:
            build_args = ["build", "--force"]
            if _use_fakeroot_build():
                build_args.append("--fakeroot")
            build_args.extend([str(sif), str(dockerfile)])
            logs = []
            proc = subprocess.Popen(
                [get_apptainer_bin()] + build_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(path),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                text = line.decode("utf-8", errors="replace")
                logs.append({"stream": text})
            if proc.wait() != 0:
                raise container_errors.BuildError(
                    f"apptainer build failed for {tag}",
                    "\n".join(entry.get("stream", "") for entry in logs),
                )
        self._write_meta(sif, [tag])
        return ApptainerImage(tags=[tag], id=sif.stem, attrs={"Created": time.time(), "Id": sif.stem}), logs

    def _write_meta(self, sif: Path, tags: List[str]) -> None:
        meta = {"tags": tags, "sif": str(sif), "created": time.time()}
        sif.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


class ApptainerContainer:
    def __init__(
        self,
        client: "ApptainerClient",
        image_ref: str,
        name: str,
        user: Optional[str] = None,
        network_mode: Optional[str] = None,
        platform: Optional[str] = None,
        nano_cpus: Optional[int] = None,
        cap_add: Optional[List[str]] = None,
        entrypoint: Optional[str] = None,
        command: Optional[Any] = None,
    ) -> None:
        self.client = client
        self.image_ref = image_ref
        self.name = name
        self.id = f"apptainer-{uuid.uuid4().hex[:12]}"
        self._user = user
        self._network_mode = network_mode
        self._platform = platform
        self._workspace = get_workspace_root() / _sanitize_tag(name)
        if self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._binds: Dict[str, Path] = {}
        self._started = False
        self._instance_name: Optional[str] = None
        self._lock = threading.Lock()

    def _sif(self) -> Path:
        return self.client.images.ensure(self.image_ref, pull=True)

    def _sandbox_dir(self) -> Optional[Path]:
        sandbox = self._workspace / "rootfs"
        if sandbox.is_dir() and (sandbox / "bin").exists():
            return sandbox
        return None

    def _writable_rootfs(self) -> Path:
        sandbox = self._sandbox_dir()
        if sandbox is not None:
            return sandbox
        sandbox = self._workspace / "rootfs"
        sif = self._sif()
        if sandbox.exists():
            shutil.rmtree(sandbox, ignore_errors=True)
        proc = _run_apptainer(
            ["build", "--sandbox", str(sandbox), str(sif)],
            timeout=self.client.timeout,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
            raise container_errors.APIError(f"apptainer sandbox build failed: {err}")
        return sandbox

    def _prepare_sandbox_bind_targets(self, sandbox: Path) -> None:
        for cpath, host in self._binds.items():
            rel = cpath.lstrip("/")
            if not rel:
                continue
            target = sandbox / rel
            if host.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.touch()
            else:
                target.mkdir(parents=True, exist_ok=True)

    def _bind_args(self) -> List[str]:
        args: List[str] = []
        for cpath, hpath in sorted(self._binds.items()):
            args.extend(["--bind", f"{hpath}:{cpath}"])
        extra = os.environ.get("APPTAINER_BINDPATH", "")
        if extra:
            for part in extra.split(","):
                part = part.strip()
                if part:
                    args.extend(["--bind", part])
        return args

    def _exec_base_args(
        self,
        workdir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        args = _exec_prefix()
        if _use_nv():
            args.append("--nv")
        if self._network_mode == "none":
            args.extend(["--network", "none"])
        elif self._network_mode == "host" and _use_host_network():
            args.extend(["--net", "--network", "host"])
        args.extend(self._bind_args())
        if workdir:
            args.extend(["--pwd", workdir])
        for key, val in (environment or {}).items():
            if val is not None:
                args.extend(["--env", f"{key}={val}"])
        args.append(str(self._writable_rootfs()))
        return args

    def _wrap_user_cmd(self, run_cmd: List[str], user: Optional[str] = None) -> List[str]:
        """Apptainer has no ``--user``. Non-root commands go through ``su``."""
        user = self._user if user is None else user
        if not user or str(user) in ("root", "0"):
            return run_cmd
        inner = " ".join(shlex.quote(str(part)) for part in run_cmd)
        return [
            "/bin/sh",
            "-c",
            f"exec su -s /bin/sh {shlex.quote(str(user))} -c {shlex.quote(inner)}",
        ]

    def start(self) -> None:
        self._started = True

    def stop(self, timeout: int = 15) -> None:
        if self._instance_name:
            try:
                _run_apptainer(["instance", "stop", self._instance_name], check=False)
            except Exception:
                pass
            self._instance_name = None

    def remove(self, force: bool = False) -> None:
        self.stop()
        registry = getattr(self.client.containers, "_registry", None)
        if isinstance(registry, dict):
            registry.pop(self.name, None)
        if self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)

    def exec_run(
        self,
        cmd: Union[str, List[str]],
        workdir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        detach: bool = False,
        **kwargs: Any,
    ) -> ExecResult:
        if isinstance(cmd, list):
            run_cmd = [str(part) for part in cmd]
        else:
            run_cmd = ["/bin/sh", "-c", cmd]
        run_cmd = self._wrap_user_cmd(run_cmd, user=kwargs.get("user", self._user))

        with self._lock:
            sandbox = self._writable_rootfs()
            self._prepare_sandbox_bind_targets(sandbox)
            base = self._exec_base_args(workdir=workdir, environment=environment)
            full_cmd = [get_apptainer_bin()] + base + run_cmd
            proc = subprocess.run(
                full_cmd,
                capture_output=True,
                timeout=self.client.timeout,
            )
        return ExecResult(exit_code=proc.returncode, output=proc.stdout + proc.stderr)

    def put_archive(self, path: str, data: bytes) -> bool:
        dest_dir = path.rstrip("/") or "/"
        with self._lock:
            sandbox = self._writable_rootfs()
            self._prepare_sandbox_bind_targets(sandbox)
            stream = io.BytesIO(data)
            with tarfile.open(fileobj=stream, mode="r") as tar:
                for member in tar.getmembers():
                    if dest_dir == "/":
                        container_path = "/" + member.name.lstrip("/")
                    else:
                        container_path = f"{dest_dir}/{member.name}".replace("//", "/")
                    use_sandbox = dest_dir not in self._binds
                    if member.isdir():
                        if use_sandbox:
                            (sandbox / container_path.lstrip("/")).mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    payload = src.read()
                    if use_sandbox or dest_dir == "/":
                        target = sandbox / container_path.lstrip("/")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(payload)
                        continue
                    host_root = self._binds[dest_dir]
                    target = host_root / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
        return True

    def get_archive(self, path: str) -> Tuple[List[bytes], dict]:
        host_path: Optional[Path] = None
        for cpath, hpath in self._binds.items():
            if path == cpath or path.startswith(cpath.rstrip("/") + "/"):
                rel = path[len(cpath) :].lstrip("/")
                host_path = hpath / rel if rel else hpath
                break
        if host_path is None or not host_path.exists():
            sandbox = self._sandbox_dir()
            if sandbox is not None:
                candidate = sandbox / path.lstrip("/")
                if candidate.exists():
                    host_path = candidate
        if host_path is None or not host_path.exists():
            raise FileNotFoundError(f"Path not found in container: {path}")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(host_path, arcname=host_path.name)
        return [buf.getvalue()], {"name": host_path.name}


class ApptainerContainers:
    def __init__(self, client: "ApptainerClient") -> None:
        self._client = client
        self._registry: Dict[str, ApptainerContainer] = {}

    def create(self, image: str, name: str, **kwargs: Any) -> ApptainerContainer:
        container = ApptainerContainer(
            self._client,
            image_ref=image,
            name=name,
            user=kwargs.get("user"),
            network_mode=kwargs.get("network_mode"),
            platform=kwargs.get("platform"),
            nano_cpus=kwargs.get("nano_cpus"),
            cap_add=kwargs.get("cap_add"),
            entrypoint=kwargs.get("entrypoint"),
            command=kwargs.get("command"),
        )
        self._registry[name] = container
        return container

    def run(self, image: str, name: str, detach: bool = True, **kwargs: Any) -> ApptainerContainer:
        container = self.create(image=image, name=name, **kwargs)
        container.start()
        return container

    def get(self, name: str) -> ApptainerContainer:
        if name in self._registry:
            return self._registry[name]
        raise container_errors.NotFound(f"Container {name} not found")

    def list(self, all: bool = False, **kwargs: Any) -> List[ApptainerContainer]:
        return list(self._registry.values())


class ApptainerClientAPI:
    def __init__(self, client: "ApptainerClient") -> None:
        self._client = client

    def build(
        self,
        path: str,
        tag: str,
        rm: bool = True,
        forcerm: bool = True,
        decode: bool = True,
        platform: Optional[str] = None,
        nocache: bool = False,
    ) -> Iterator[dict]:
        try:
            _image, logs = self._client.images.build(
                path=path, tag=tag, rm=rm, nocache=nocache
            )
        except container_errors.BuildError as exc:
            # docker-py's low-level build API yields a log stream, then an
            # errorDetail chunk. polyglot/docker_build.py writes those chunks
            # to build_image.log; a raised BuildError alone only keeps str(exc).
            yield from _build_error_stream(exc)
            return
        yield from logs

    def inspect_container(self, container_id: str) -> dict:
        return {"State": {"Pid": 0}}


class ApptainerClient:
    """DockerClient-compatible facade using Apptainer."""

    def __init__(self, timeout: Optional[int] = None) -> None:
        if timeout is None:
            timeout = int(
                os.environ.get(
                    "APPTAINER_API_TIMEOUT",
                    os.environ.get("CONTAINER_API_TIMEOUT", "600"),
                )
            )
        self.timeout = timeout
        get_image_dir().mkdir(parents=True, exist_ok=True)
        get_workspace_root().mkdir(parents=True, exist_ok=True)
        self.images = ApptainerImages(self)
        self.containers = ApptainerContainers(self)
        self.api = ApptainerClientAPI(self)

    def ping(self) -> bool:
        proc = _run_apptainer(["version"], timeout=30, check=False)
        return proc.returncode == 0

    def info(self) -> dict:
        proc = _run_apptainer(["version"], timeout=30, check=False)
        version = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        first = version.splitlines()[0] if version else "unknown"
        return {"Name": f"apptainer ({first})"}

    def version(self) -> dict:
        proc = _run_apptainer(["version"], timeout=30, check=False)
        text = (proc.stdout or b"").decode("utf-8", errors="replace")
        for line in text.splitlines():
            if "apptainer version" in line.lower() or "singularity version" in line.lower():
                return {"Version": line.split()[-1]}
        return {"Version": "unknown"}
