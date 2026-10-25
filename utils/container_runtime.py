"""Select Docker or Apptainer without changing docker-py call sites.

``DGM_CONTAINER_RUNTIME`` (alias ``CONTAINER_RUNTIME``):

- ``auto`` (default): use Docker when ``docker info`` succeeds, otherwise
  Apptainer/Singularity when that binary is on ``PATH``.
- ``docker``: always ``docker.from_env``.
- ``apptainer`` or ``singularity``: always :class:`utils.apptainer_compat.ApptainerClient`.

Only the ``auto`` decision is cached. Call :func:`clear_runtime_cache` (or
``resolve_container_runtime(refresh=True)``) after changing the environment in
tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

_AUTO_DECISION: Optional[str] = None


def clear_runtime_cache() -> None:
    global _AUTO_DECISION
    _AUTO_DECISION = None


def find_apptainer_bin() -> Optional[str]:
    """Return an executable Apptainer or Singularity binary, if one exists."""
    explicit = os.environ.get("APPTAINER_BIN", "").strip()
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        return shutil.which(explicit)
    return shutil.which("apptainer") or shutil.which("singularity")


def docker_daemon_available() -> bool:
    """True when the docker CLI can talk to a daemon.

    The probe is intentionally short (``DGM_DOCKER_PROBE_TIMEOUT``, default 3s)
    so a stale ``DOCKER_HOST`` does not block HPC login nodes.
    """
    if shutil.which("docker") is None:
        return False
    timeout = float(os.environ.get("DGM_DOCKER_PROBE_TIMEOUT", "3"))
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def resolve_container_runtime(refresh: bool = False) -> str:
    """Return ``docker`` or ``apptainer``."""
    global _AUTO_DECISION
    explicit = (
        os.environ.get("DGM_CONTAINER_RUNTIME")
        or os.environ.get("CONTAINER_RUNTIME")
        or "auto"
    ).strip().lower()
    if explicit == "singularity":
        return "apptainer"
    if explicit in ("docker", "apptainer"):
        return explicit
    if explicit != "auto":
        raise RuntimeError(
            f"Unknown container runtime {explicit!r}. "
            "Use docker, apptainer, singularity, or auto."
        )
    if refresh:
        _AUTO_DECISION = None
    if _AUTO_DECISION:
        return _AUTO_DECISION
    if docker_daemon_available():
        _AUTO_DECISION = "docker"
        return "docker"
    if find_apptainer_bin():
        _AUTO_DECISION = "apptainer"
        return "apptainer"
    raise RuntimeError(
        "No container runtime available. Start a Docker daemon, or install "
        "Apptainer and set DGM_CONTAINER_RUNTIME=apptainer (or leave it as "
        "auto once `apptainer` is on PATH)."
    )


def container_from_env(timeout: Optional[int] = None):
    """Return a docker-py client or an Apptainer facade with the same shape."""
    runtime = resolve_container_runtime()
    if runtime == "apptainer":
        bin_path = find_apptainer_bin()
        if not bin_path:
            raise RuntimeError(
                "Container runtime is apptainer, but neither `apptainer` nor "
                "`singularity` is executable. Install Apptainer or set APPTAINER_BIN."
            )
        os.environ["APPTAINER_BIN"] = bin_path
        from utils.apptainer_compat import ApptainerClient

        return ApptainerClient(timeout=timeout)

    import docker

    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return docker.from_env(**kwargs)
