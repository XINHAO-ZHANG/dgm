"""SWE-bench grading via Apptainer (docker-py shim, no Docker daemon).

Patches ``docker.from_env`` in this process, then runs
``swebench.harness.run_evaluation``. Env and base images are still built
through the shim when they are missing. This does not skip those builds and
does not assume pre-pulled Epoch SIFs.

Requires the ``swebench`` package (see README: clone SWE-bench and
``pip install -e .``). The ``docker`` Python package may be installed; it is
only used as the module whose ``from_env`` we replace.
"""

from __future__ import annotations

import shlex
import time


def _patch_docker_from_env() -> None:
    import docker as docker_mod

    from utils.docker_utils import docker_from_env, get_container_api_timeout

    def _from_env(timeout=None):
        if timeout is None:
            timeout = get_container_api_timeout()
        return docker_from_env(timeout=timeout)

    docker_mod.from_env = _from_env


def apptainer_exec_run_with_timeout(container, cmd, timeout: int | None = 60):
    """Upstream SWE-bench uses ``exec_create``. Apptainer uses ``timeout(1)``."""
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(str(part)) for part in cmd)
    else:
        cmd_str = str(cmd)
    if timeout is not None and timeout > 0:
        cmd_str = f"timeout {int(timeout)} {cmd_str}"
    start = time.time()
    result = container.exec_run(cmd_str, workdir="/")
    output = result.output if isinstance(result.output, bytes) else b""
    text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
    timed_out = result.exit_code == 124
    return text, timed_out, time.time() - start


def patch_swebench_for_apptainer() -> None:
    """Patch docker-py entry points used by the upstream harness.

    Import ``swebench`` before calling this. Image builds stay enabled.
    """
    import swebench.harness.docker_utils as docker_utils

    _patch_docker_from_env()
    docker_utils.exec_run_with_timeout = apptainer_exec_run_with_timeout


def main() -> None:
    _patch_docker_from_env()
    try:
        patch_swebench_for_apptainer()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "swebench is not installed. Clone SWE-bench and pip install -e . "
            "before grading. See APPTAINER.md."
        ) from exc

    import runpy

    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
