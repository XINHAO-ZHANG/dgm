# Apptainer runtime defaults for DGM jobs on HPC login/compute nodes.
# Source from the repo root:
#   . "${REPO_ROOT}/scripts/slurm/apptainer_runtime.inc.sh"
#
# Does not require a Docker daemon, DOCKER_HOST, or ENABLE_REMOTE_DOCKER.
# User-local Apptainer (for example ~/software/apptainer/bin) must already be
# on PATH before sourcing this file.

_DGM_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export DGM_CONTAINER_RUNTIME="${DGM_CONTAINER_RUNTIME:-apptainer}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${_DGM_ROOT}/apptainer_images}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _APPTAINER_WS_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
    export APPTAINER_WORKSPACE_ROOT="${APPTAINER_WORKSPACE_ROOT:-${_APPTAINER_WS_BASE}/apptainer-workspaces-${SLURM_JOB_ID}}"
else
    export APPTAINER_WORKSPACE_ROOT="${APPTAINER_WORKSPACE_ROOT:-${TMPDIR:-/tmp}/apptainer-workspaces}"
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_IMAGE_DIR}/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}"

# --net --network host often needs fakeroot or root on HPC. Leave it off.
# Unprivileged Apptainer already shares the host network, which pip and API
# calls need. Do not pass --network none.
export APPTAINER_USE_HOST_NETWORK="${APPTAINER_USE_HOST_NETWORK:-0}"
export APPTAINER_BUILD_FAKEROOT="${APPTAINER_BUILD_FAKEROOT:-1}"
export APPTAINER_CONTAIN="${APPTAINER_CONTAIN:-1}"
export APPTAINER_NO_MOUNT="${APPTAINER_NO_MOUNT:-tmp}"
# pb.base (apt+go+rust+npm+conda, then mksquashfs) exceeds the 600s exec default.
export APPTAINER_BUILD_TIMEOUT="${APPTAINER_BUILD_TIMEOUT:-7200}"
export APPTAINER_API_TIMEOUT="${APPTAINER_API_TIMEOUT:-7200}"

mkdir -p "${APPTAINER_IMAGE_DIR}" "${APPTAINER_WORKSPACE_ROOT}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

apptainer_runtime_verify() {
    if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
        echo "ERROR: apptainer/singularity not found in PATH." >&2
        echo "  Example: export PATH=\"\$HOME/software/apptainer/bin:\$PATH\"" >&2
        return 1
    fi
    if command -v apptainer >/dev/null 2>&1; then
        echo "Apptainer runtime: $(apptainer version 2>/dev/null | head -1)"
    else
        echo "Singularity runtime: $(singularity version 2>/dev/null | head -1)"
    fi
    echo "  DGM_CONTAINER_RUNTIME=${DGM_CONTAINER_RUNTIME}"
    echo "  APPTAINER_IMAGE_DIR=${APPTAINER_IMAGE_DIR}"
    echo "  APPTAINER_WORKSPACE_ROOT=${APPTAINER_WORKSPACE_ROOT}"
    echo "  APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
    return 0
}

unset _DGM_ROOT _APPTAINER_WS_BASE
