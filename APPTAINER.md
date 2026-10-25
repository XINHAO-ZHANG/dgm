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
# Client pull/exec timeout. Dockerfile builds default to 7200s on their own;
# set this as well when you want every Apptainer call on that budget.
export APPTAINER_API_TIMEOUT=7200
export APPTAINER_BUILD_TIMEOUT=7200

mkdir -p "$APPTAINER_IMAGE_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_WORKSPACE_ROOT"
```

Slurm jobs can source `scripts/slurm/apptainer_runtime.inc.sh`, which sets the same variables (image dir defaults to `./apptainer_images` in the repo, workspace under `$SLURM_TMPDIR` when present).

## Fakeroot during image builds

`APPTAINER_BUILD_FAKEROOT=1` (the default) passes `--fakeroot` to two build steps:

- the Dockerfile post-script `apptainer exec` that runs `RUN` (apt, conda, rustup, pip)
- the final `apptainer build` that packs the sandbox into a SIF

`pb.base.x86_64` installs packages as root. Without `--fakeroot` on that exec, apt fails and the build raises `apptainer Dockerfile post script failed`. The post-script stdout and stderr are included in the `BuildError` text and in the docker-py build stream (`build_image.log`).

bach4 has no `/etc/subuid` entry. `apptainer exec --fakeroot` still runs there through Apptainer's root-mapped user namespace: `apptainer exec --fakeroot docker://ubuntu:22.04 id` prints `uid=0`. A missing subuid map is not a reason to disable fakeroot.

Fakeroot also needs libraries inside the image. `ubuntu:22.04` and `buildpack-deps:jammy` (the Polyglot base) work. Alpine does not: `apptainer exec --fakeroot docker://alpine` fails because the image has no fakeroot support library. Keep jammy/Ubuntu bases for images whose `RUN` steps need root.

If fakeroot itself errors, turn it off:

```bash
export APPTAINER_BUILD_FAKEROOT=0
```

Root package installs in `RUN` then fail unless the base image already contains them.

## Build timeout

`pb.base.x86_64:latest` runs apt, Go, Rust, npm, and conda in one post script, then `apptainer build --fakeroot` (mksquashfs). That does not finish in 600 seconds. On bach4 the 600s client timeout stopped the post script mid-npm. The same build with a 7200s limit got through conda and into mksquashfs.

Dockerfile builds (sandbox creation, the post script, and the final SIF pack) default to **7200** seconds. `docker_from_env()` still uses **600** seconds for pull and `exec_run` unless you override it. Precedence for the build budget:

| Variable | Effect |
| --- | --- |
| `APPTAINER_BUILD_TIMEOUT` | Dockerfile build budget. Recommended **7200** for a full Polyglot base image. |
| `APPTAINER_API_TIMEOUT` or `CONTAINER_API_TIMEOUT` | Used for builds only when `APPTAINER_BUILD_TIMEOUT` is unset. Also the pull/exec client timeout (default 600 when unset). |
| neither set | Builds use 7200. Pull and exec stay at 600. |

```bash
export APPTAINER_BUILD_TIMEOUT=7200
```

Smoke checks (`ping`, `alpine echo`) do not need this. Set it before Polyglot builds `pb.base.*`.

Unprivileged Apptainer already shares the host network, which `pip` and model API calls need. `APPTAINER_USE_HOST_NETWORK=1` adds `--net --network host` and often needs fakeroot; leave it at `0`.

`APPTAINER_CONTAIN=1` and `APPTAINER_NO_MOUNT=tmp` (the defaults) apply to **running** containers. They stop Apptainer from hiding the sandbox `/tmp` and image home with the host's `/tmp` and `$HOME`. Polyglot writes the patch to `/tmp/patch.diff` inside the container. Explicit `APPTAINER_BINDPATH` binds are still applied.

The Dockerfile post script does **not** use `--contain`. With `--fakeroot`, that flag mounts an empty directory over `/root` and hides files `COPY` just placed there. `pb.env` then fails with `chmod: cannot access '/root/setup_env.sh'`. The post script uses `--writable`, `--no-home`, and `--no-mount tmp,home,cwd` so sandbox `/root` (conda, rustup, `setup_env.sh`) and `/tmp` stay visible. `APPTAINER_CONTAIN` does not change that build step.

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

A full Polyglot base build (`pb.base.x86_64:latest`, apt+go+rust+npm+conda, then mksquashfs) needs `APPTAINER_BUILD_TIMEOUT=7200`. That is the Dockerfile-build default; the Alpine smoke above does not use it.

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
