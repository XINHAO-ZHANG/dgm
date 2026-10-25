"""Check the selected container runtime without starting a full evaluation.

Default mode only calls ``ping()`` / ``info()`` and does not pull images.
``--exec`` pulls ``alpine:latest`` and runs ``echo ok`` inside a sandbox.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exec",
        action="store_true",
        help="Pull alpine:latest and exec 'echo ok' (needs network and Apptainer or Docker).",
    )
    args = parser.parse_args(argv)

    from utils.container_runtime import resolve_container_runtime
    from utils.docker_utils import docker_from_env

    runtime = resolve_container_runtime()
    client = docker_from_env()
    print(f"runtime={runtime}")
    print(f"ping={client.ping()}")
    print(f"info={client.info()}")
    if not args.exec:
        return 0

    image = "alpine:latest"
    print(f"pulling {image}")
    client.images.pull(image)
    container = client.containers.create(image=image, name="dgm-container-smoke", command="echo ok")
    try:
        container.start()
        result = container.exec_run("echo ok")
        text = result.output.decode() if isinstance(result.output, bytes) else result.output
        print(f"exit={result.exit_code}")
        print(text)
        return 0 if result.exit_code == 0 else 1
    finally:
        container.remove()


if __name__ == "__main__":
    sys.exit(main())
