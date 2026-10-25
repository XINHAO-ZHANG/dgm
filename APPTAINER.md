# Apptainer on HPC (no Docker daemon)

DGM's evaluation and self-improve paths talk to containers through the docker-py shape (`images.build`, `containers.create`, `exec_run`, `put_archive`). `utils.docker_utils.docker_from_env` returns either a real Docker client or `ApptainerClient`.

This is the same layout as [MGM](https://github.com/RealLcz/MGM): `apptainer pull` writes a `.sif`, `apptainer build --sandbox` makes a writable rootfs, and `apptainer exec --writable` runs commands. There is no third container API, and `ENABLE_REMOTE_DOCKER` / remote SSH Docker are not required.

## Which runtime is used

`DGM_CONTAINER_RUNTIME` (alias `CONTAINER_RUNTIME`):

| Value | Behavior |
| --- | --- |
| `auto` (default) | `docker info` with a short timeout (`DGM_DOCKER_PROBE_TIMEOUT`, default 3s). If the daemon answers, use Docker. Otherwise, if `apptainer` or `singularity` is on `PATH` (or `APPTAINER_BIN`), use Apptainer. |
| `docker` | Always `docker.from_env`. A dead `DOCKER_HOST` is not rewritten. |
| `apptainer` or `singularity` | Always Apptainer. |

The Python package `docker` can stay installed. It does not start a daemon. On the Apptainer path its exception types are reused so existing `except docker.errors.ImageNotFound` / `BuildError` handlers still match.

## bach4-style setup

User-local Apptainer 1.4.x, no root and no setuid mount helper:

```bash
export PATH="$HOME/software/apptainer/bin:$PATH"
export DGM_CONTAINER_RUNTIME=apptainer

# Optional overrides. Defaults are under $HF_HOME/apptainer_images or
# ~/.cache/huggingface/apptainer_images, and $TMPDIR/apptainer-workspaces.
export APPTAINER_IMAGE_DIR="$HOME/.cache/apptainer/images"
export APPTAINER_CACHEDIR="$APPTAINER_IMAGE_DIR/cache"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"
export APPTAINER_WORKSPACE_ROOT="${TMPDIR:-/tmp}/apptainer-workspaces"
export APPTAINER_API_TIMEOUT=7200

mkdir -p "$APPTAINER_IMAGE_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_WORKSPACE_ROOT"
```

Slurm jobs can source `scripts/slurm/apptainer_runtime.inc.sh`, which sets the same variables (image dir defaults to `./apptainer_images` in the repo, workspace under `$SLURM_TMPDIR` when present).

If `--fakeroot` is unavailable (no subuid mapping), build with:

```bash
export APPTAINER_BUILD_FAKEROOT=0
```

Unprivileged Apptainer already shares the host network, which `pip` and model API calls need. `APPTAINER_USE_HOST_NETWORK=1` adds `--net --network host` and often needs fakeroot; leave it at `0`.

`APPTAINER_CONTAIN=1` and `APPTAINER_NO_MOUNT=tmp` (the defaults) stop Apptainer from hiding the sandbox `/tmp` and image home with the host's `/tmp` and `$HOME`. Polyglot writes the patch to `/tmp/patch.diff` inside the container. Explicit `APPTAINER_BINDPATH` binds are still applied.

## Smoke test

No dataset and no API key:

```bash
python -c "from utils.docker_utils import docker_from_env; c=docker_from_env(); print(c.ping(), c.info())"
```

Pull Alpine and exec (needs network, writes a SIF):

```bash
python scripts/container_smoke.py --exec
# equivalent to: apptainer exec docker://alpine:latest echo ok
```

## What runs without a Docker daemon

| Entrypoint | Role |
| --- | --- |
| `polyglot/harness.py` | Polyglot agent evaluation. Builds `pb.base.*` / `pb.env.*` / `pb.eval.*` through the shim. |
| `polyglot/run_evaluation.py` | Polyglot grading. `exec_run_with_timeout` uses `timeout(1)` inside the sandbox (exit 124 is a timeout). |
| `swe_bench/harness.py` | SWE-bench agent evaluation, in-process. The upstream `swebench` package must be installed. The client passed into `build_container` / `build_env_images` is the shim. |
| `swe_bench/report.py` | SWE-bench grading. On Apptainer this spawns `python -m swe_bench.run_evaluation_apptainer`, which patches `docker.from_env` in the child before `swebench.harness.run_evaluation`. Env and base images are still built; they are not skipped. |
| `self_improve_step.py` / `DGM_outer.py` | Builds the `dgm` image from the repo `Dockerfile` (`COPY . .` into `/dgm`) and runs the agent inside that sandbox. The host checkout is not bind-mounted over `/dgm`. |

The first real run still pulls or builds images. That work is heavy; the shim only removes the need for `dockerd`.

`apptainer/dgm.def` is a manual base image (Python 3.10, git, `/dgm`) for `apptainer build`. Self-improve uses the Dockerfile so the repo contents and `.dockerignore` match the Docker path.

## Image builds

- Local tags such as `pb.base.*` and `dgm` are stored as `{sanitized_repo}_{tag}.sif` plus a `.json` sidecar (the tag list and a `created` timestamp).
- `sweb.eval.*` / `sweb.base.*` / `sweb.env.*` pull from `ghcr.io/epoch-research/swe-bench...` (`SWE_GHCR_EPOCH_PREFIX`). A Dockerfile `FROM` prefers an already-built local SIF with that tag, so `pb.env.*` is layered on `pb.base.*`.
- `images.get` does not invent a missing `sweb.env.*` image. The image is built or pulled.
- `images.remove` keeps `sweb.eval.*` unless `APPTAINER_REMOVE_EVAL_IMAGES=1`, so a cleanup pass does not delete an expensive pre-pulled eval SIF. `pb.eval.*` can be removed.
- Rebuild an existing SIF by building with `nocache=True`. `build_dgm_container(..., force_rebuild=True)` does that on the Apptainer path only.

## Limits

The Dockerfile translator covers `FROM` (including `--platform`), `WORKDIR`, `COPY`, `ENV`, `RUN` (backslash continuations), and `ADD` of an `http(s)` URL. `COPY` steps are applied before `RUN` steps. That matches the Dockerfiles in this repo (apt, then copy, then pip / setup scripts). A `RUN` that creates files a later `COPY` must overwrite is not expressed.

GPU: `APPTAINER_NV=1` adds `--nv`.

Unit tests under `tests/test_apptainer_compat.py` mock `subprocess` and do not need a daemon or a real `apptainer` binary.
