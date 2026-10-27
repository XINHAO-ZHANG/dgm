"""Host disk and memory gates for the DGM and EvoPilot outer loops.

A shared machine can fill ``/tmp`` with Apptainer sandboxes or drive
memory pressure high enough that user lookup (LDAP/sssd) dies. These
checks run before a generation calls the model and before Verify starts
another sandbox. They do not decide whether a patch is good.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from utils.secret_redaction import redact_secrets


OWNER_ENV = "DGM_WORKSPACE_OWNER"
MARKER_NAME = ".dgm-workspace-owner"
MIN_FREE_ENV = "DGM_MIN_TMP_FREE_BYTES"
MIN_MEM_ENV = "DGM_MIN_AVAILABLE_MEM_BYTES"
PRESSURE_ENV = "DGM_MAX_MEM_PRESSURE"

# Headroom for one more generation on a shared host, not a reservation
# for every parallel sandbox. Raise the CLI values when several eval
# workers or two outer loops share the same /tmp.
DEFAULT_MIN_TMP_FREE_GIB = 10.0
DEFAULT_MIN_AVAILABLE_MEM_GIB = 4.0
DEFAULT_MAX_MEM_PRESSURE_AVG10 = 10.0

DISK_CODE = "resource_exhausted_disk"
MEMORY_CODE = "resource_exhausted_memory"
PRESSURE_CODE = "resource_exhausted_memory_pressure"
INFRA_NOSPACE = "infra_nospace"
INFRA_MEMORY = "infra_memory"
INFRA_LDAP = "infra_ldap_userid"
INFRA_APPTAINER = "infra_apptainer_start"
INFRA_UNHANDLED = "infra_unhandled"

_GIB = 1024 ** 3
_PROCESS_OWNER = f"pid-{os.getpid()}"
_OWNED_PATHS: set[str] = set()
_OWNED_LOCK = threading.Lock()

_NOSPACE = re.compile(
    r"no space left on device|errno\s*28\b|disk quota exceeded|edquot",
    re.IGNORECASE,
)
_MEMORY = re.compile(
    r"cannot allocate memory|out of memory|memoryerror|errno\s*12\b|oom-kill",
    re.IGNORECASE,
)
_LDAP = re.compile(
    r"failed to get (?:user|uid|gid)|unknown userid|getpwuid|getpwnam|"
    r"user(?:id)? lookup|nss_ldap|sssd|"
    r"could not (?:get|lookup) (?:pwuid|user)",
    re.IGNORECASE,
)
_APPTAINER_START = re.compile(
    r"apptainer sandbox build failed|"
    r"failed to (?:start|create) (?:the )?(?:apptainer )?container|"
    r"instance start failed|"
    r"failed to (?:create|spawn) (?:apptainer )?instance|"
    r"apptainer: not found|executable file `apptainer` not found",
    re.IGNORECASE,
)


class ResourceGateError(RuntimeError):
    """Free space or memory is below the configured floor."""

    def __init__(self, code: str, message: str, snapshot: "ResourceSnapshot | None" = None):
        self.code = code
        self.snapshot = snapshot
        super().__init__(message)


class InfraEvalError(RuntimeError):
    """Verify failed because the host or Apptainer could not run."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ResourceThresholds:
    min_free_bytes: int = int(DEFAULT_MIN_TMP_FREE_GIB * _GIB)
    min_available_memory_bytes: int = int(DEFAULT_MIN_AVAILABLE_MEM_GIB * _GIB)
    max_memory_pressure_avg10: float | None = DEFAULT_MAX_MEM_PRESSURE_AVG10

    def to_dict(self) -> dict:
        return {
            "min_free_bytes": self.min_free_bytes,
            "min_available_memory_bytes": self.min_available_memory_bytes,
            "max_memory_pressure_avg10": self.max_memory_pressure_avg10,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    paths: dict[str, int | None]
    mem_available_bytes: int | None
    psi_full_avg10: float | None
    psi_some_avg10: float | None

    def to_dict(self) -> dict:
        return {
            "paths": dict(self.paths),
            "mem_available_bytes": self.mem_available_bytes,
            "psi_full_avg10": self.psi_full_avg10,
            "psi_some_avg10": self.psi_some_avg10,
        }


def gib_to_bytes(value: float) -> int:
    """Return 0 when ``value`` is not positive so that check is disabled."""

    if value <= 0:
        return 0
    return int(value * _GIB)


def format_bytes(n: int) -> str:
    if abs(n) >= _GIB:
        return f"{n / _GIB:.2f} GiB"
    return f"{n / (1024 ** 2):.1f} MiB"


def add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``DGM_outer`` and ``EvoPilot_outer``."""

    parser.add_argument(
        "--min-tmp-free-gb",
        type=float,
        default=DEFAULT_MIN_TMP_FREE_GIB,
        help=(
            "Refuse Improve and Verify when /tmp or the Apptainer workspace root "
            f"has less than this many GiB free (default {DEFAULT_MIN_TMP_FREE_GIB:g}). "
            "0 disables the disk check."
        ),
    )
    parser.add_argument(
        "--min-available-mem-gb",
        type=float,
        default=DEFAULT_MIN_AVAILABLE_MEM_GIB,
        help=(
            "Refuse Improve and Verify when MemAvailable is below this many GiB "
            f"(default {DEFAULT_MIN_AVAILABLE_MEM_GIB:g}). 0 disables the memory check."
        ),
    )
    parser.add_argument(
        "--max-mem-pressure",
        type=float,
        default=DEFAULT_MAX_MEM_PRESSURE_AVG10,
        help=(
            "Refuse Improve and Verify when /proc/pressure/memory full avg10 "
            f"is above this percent (default {DEFAULT_MAX_MEM_PRESSURE_AVG10:g}). "
            "A negative value disables the PSI check. Missing PSI is ignored."
        ),
    )
    parser.add_argument(
        "--continue-on-infra-error",
        action="store_true",
        help=(
            "After a nospace, memory, LDAP/userid, or Apptainer start failure, "
            "keep going to the next generation. The default stops the outer loop. "
            "This does not bypass the disk and memory gates."
        ),
    )
    parser.add_argument(
        "--dry-resource-check",
        action="store_true",
        help="Print the host snapshot and whether the gates would refuse a generation, then exit. No model call and no sandbox.",
    )


def thresholds_from_args(args: argparse.Namespace) -> ResourceThresholds:
    pressure = float(getattr(args, "max_mem_pressure", DEFAULT_MAX_MEM_PRESSURE_AVG10))
    return ResourceThresholds(
        min_free_bytes=gib_to_bytes(float(getattr(args, "min_tmp_free_gb", DEFAULT_MIN_TMP_FREE_GIB))),
        min_available_memory_bytes=gib_to_bytes(
            float(getattr(args, "min_available_mem_gb", DEFAULT_MIN_AVAILABLE_MEM_GIB))
        ),
        max_memory_pressure_avg10=None if pressure < 0 else pressure,
    )


def publish_resource_policy(thresholds: ResourceThresholds) -> None:
    """Expose the outer-loop thresholds to self-improve and the harness."""

    os.environ[MIN_FREE_ENV] = str(int(thresholds.min_free_bytes))
    os.environ[MIN_MEM_ENV] = str(int(thresholds.min_available_memory_bytes))
    if thresholds.max_memory_pressure_avg10 is None:
        os.environ[PRESSURE_ENV] = "-1"
    else:
        os.environ[PRESSURE_ENV] = str(thresholds.max_memory_pressure_avg10)


def thresholds_from_env() -> ResourceThresholds:
    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return default
        return int(float(raw))

    pressure_raw = os.environ.get(PRESSURE_ENV, "").strip()
    if pressure_raw == "":
        pressure: float | None = DEFAULT_MAX_MEM_PRESSURE_AVG10
    else:
        parsed = float(pressure_raw)
        pressure = None if parsed < 0 else parsed
    return ResourceThresholds(
        min_free_bytes=max(0, _int(MIN_FREE_ENV, int(DEFAULT_MIN_TMP_FREE_GIB * _GIB))),
        min_available_memory_bytes=max(0, _int(MIN_MEM_ENV, int(DEFAULT_MIN_AVAILABLE_MEM_GIB * _GIB))),
        max_memory_pressure_avg10=pressure,
    )


def sanitize_owner(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned[:120] or "dgm-run"


def current_workspace_owner() -> str:
    raw = os.environ.get(OWNER_ENV, "").strip()
    if raw:
        return sanitize_owner(raw)
    return _PROCESS_OWNER


def ensure_workspace_owner(output_dir: str) -> str:
    """Stable owner token for this output directory, including resumes."""

    path = os.path.join(output_dir, ".dgm_workspace_owner")
    if os.path.isfile(path):
        owner = sanitize_owner(Path(path).read_text(encoding="utf-8"))
    else:
        base = sanitize_owner(os.path.basename(os.path.abspath(output_dir)))
        owner = sanitize_owner(f"dgm-{base}")
        os.makedirs(output_dir, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(owner + "\n")
    os.environ[OWNER_ENV] = owner
    return owner


def note_owned_workspace(path: os.PathLike) -> None:
    with _OWNED_LOCK:
        _OWNED_PATHS.add(os.path.abspath(str(path)))


def stamp_workspace(path: os.PathLike, name: str = "") -> None:
    """Mark a sandbox directory as belonging to this run.

    Other users' directories have no marker, or a different owner, and
    cleanup leaves them alone. The marker stores the owner token and the
    container name, never environment values.
    """

    directory = Path(path)
    note_owned_workspace(directory)
    owner = current_workspace_owner()
    safe_name = sanitize_owner(name) if name else ""
    payload = f"owner={owner}\nname={safe_name}\n"
    marker = directory / MARKER_NAME
    try:
        marker.write_text(payload, encoding="utf-8")
        os.chmod(marker, 0o644)
    except OSError:
        return


def read_owner_marker(path: os.PathLike) -> str | None:
    marker = Path(path) / MARKER_NAME
    try:
        text = marker.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("owner="):
            token = line.split("=", 1)[1].strip()
            return token or None
    return None


def _workspace_root() -> Path:
    from utils.apptainer_compat import get_workspace_root

    return get_workspace_root()


def cleanup_owned_workspaces(
    root: os.PathLike | None = None,
    owner: str | None = None,
) -> list[str]:
    """Remove this run's Apptainer workspace directories.

    Only direct children of the workspace root are considered. A directory
    is removed when its marker owner matches, or when this process created
    it and it has no marker. Symlinks and every other owner's directory stay.
    """

    base = Path(root) if root is not None else _workspace_root()
    token = sanitize_owner(owner) if owner else current_workspace_owner()
    if not token or not base.is_dir():
        return []
    base_resolved = base.resolve()
    with _OWNED_LOCK:
        noted = set(_OWNED_PATHS)
    removed: list[str] = []
    try:
        children = list(base.iterdir())
    except OSError:
        return []
    for child in children:
        try:
            if child.is_symlink() or not child.is_dir():
                continue
            # Stay inside the workspace root even if a name is surprising.
            if base_resolved not in child.resolve().parents:
                continue
        except OSError:
            continue
        stamped = read_owner_marker(child)
        noted_here = os.path.abspath(str(child)) in noted or str(child.resolve()) in noted
        if stamped is not None:
            if stamped != token:
                continue
        elif not noted_here:
            continue
        shutil.rmtree(child, ignore_errors=True)
        if not child.exists():
            removed.append(str(child))
            with _OWNED_LOCK:
                _OWNED_PATHS.discard(os.path.abspath(str(child)))
    return removed


def _existing_anchor(path: str) -> str | None:
    current = os.path.abspath(path)
    while not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
    return current


def paths_to_check() -> list[str]:
    """``/tmp`` plus any configured temp or Apptainer workspace root."""

    requested = ["/tmp"]
    for key in ("TMPDIR", "APPTAINER_TMPDIR", "APPTAINER_WORKSPACE_ROOT"):
        value = os.environ.get(key, "").strip()
        if value:
            requested.append(value)
    labels: list[str] = []
    seen: set[str] = set()
    for item in requested:
        anchor = _existing_anchor(item)
        if anchor is None:
            continue
        label = item if os.path.exists(item) else anchor
        identity = os.path.realpath(label)
        if identity in seen:
            continue
        seen.add(identity)
        labels.append(label)
    return labels


def _read_mem_available(text: str) -> int | None:
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].endswith(":"):
            continue
        if not parts[1].isdigit():
            continue
        # /proc/meminfo reports kB.
        values[parts[0][:-1]] = int(parts[1]) * 1024
    if "MemAvailable" in values:
        return values["MemAvailable"]
    if "MemFree" in values:
        return values["MemFree"]
    return None


def _read_psi(text: str) -> tuple[float | None, float | None]:
    some = None
    full = None
    for line in text.splitlines():
        kind = "some" if line.startswith("some") else "full" if line.startswith("full") else ""
        if not kind:
            continue
        for part in line.split():
            if part.startswith("avg10="):
                try:
                    value = float(part.split("=", 1)[1])
                except ValueError:
                    value = None
                if kind == "some":
                    some = value
                else:
                    full = value
    return some, full


def snapshot_resources(paths: list[str] | None = None) -> ResourceSnapshot:
    chosen = paths if paths is not None else paths_to_check()
    free: dict[str, int | None] = {}
    for label in chosen:
        anchor = _existing_anchor(label)
        if anchor is None:
            free[label] = None
            continue
        try:
            free[label] = int(shutil.disk_usage(anchor).free)
        except OSError:
            free[label] = None
    mem_available = None
    try:
        mem_available = _read_mem_available(Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        mem_available = None
    some = None
    full = None
    try:
        some, full = _read_psi(Path("/proc/pressure/memory").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        some, full = None, None
    return ResourceSnapshot(
        paths=free,
        mem_available_bytes=mem_available,
        psi_full_avg10=full,
        psi_some_avg10=some,
    )


def assess_resources(
    thresholds: ResourceThresholds,
    snapshot: ResourceSnapshot,
) -> tuple[str | None, str]:
    """Return ``(code, detail)``. ``code`` is None when the host may proceed."""

    if thresholds.min_free_bytes > 0:
        short = []
        measured = False
        for label, free in snapshot.paths.items():
            if free is None:
                continue
            measured = True
            if free < thresholds.min_free_bytes:
                short.append(f"{label} has {format_bytes(free)} free")
        if short:
            detail = (
                f"{'; '.join(short)}; need at least {format_bytes(thresholds.min_free_bytes)}"
            )
            return DISK_CODE, detail
        if snapshot.paths and not measured:
            return DISK_CODE, "could not measure free space on /tmp or the workspace root"

    if thresholds.min_available_memory_bytes > 0:
        mem = snapshot.mem_available_bytes
        if mem is None:
            return MEMORY_CODE, "could not read MemAvailable"
        if mem < thresholds.min_available_memory_bytes:
            return (
                MEMORY_CODE,
                f"MemAvailable is {format_bytes(mem)}; need at least "
                f"{format_bytes(thresholds.min_available_memory_bytes)}",
            )

    limit = thresholds.max_memory_pressure_avg10
    if limit is not None and snapshot.psi_full_avg10 is not None and snapshot.psi_full_avg10 > limit:
        return (
            PRESSURE_CODE,
            f"memory pressure full avg10 is {snapshot.psi_full_avg10:.1f}; limit is {limit:.1f}",
        )
    return None, "ok"


def require_resources(
    thresholds: ResourceThresholds | None = None,
    *,
    phase: str,
    probe=None,
) -> ResourceSnapshot:
    """Raise ``ResourceGateError`` before a model call or a new sandbox."""

    chosen = thresholds if thresholds is not None else thresholds_from_env()
    snapshot = probe() if probe is not None else snapshot_resources()
    code, detail = assess_resources(chosen, snapshot)
    if code:
        message = (
            f"refusing {phase}: {detail}. "
            "No model call and no new Apptainer sandbox will be started."
        )
        raise ResourceGateError(code, message, snapshot)
    return snapshot


def require_resources_from_env(*, phase: str) -> ResourceSnapshot:
    return require_resources(thresholds_from_env(), phase=phase)


def resource_check_report(thresholds: ResourceThresholds | None = None) -> dict:
    chosen = thresholds if thresholds is not None else thresholds_from_env()
    snapshot = snapshot_resources()
    code, detail = assess_resources(chosen, snapshot)
    message = detail if code is None else (
        f"refusing improve: {detail}. "
        "No model call and no new Apptainer sandbox will be started."
    )
    return {
        "ok": code is None,
        "code": code,
        "message": message,
        "thresholds": chosen.to_dict(),
        "snapshot": snapshot.to_dict(),
    }


def classify_infra_error(text: object) -> str | None:
    """Map a host or Apptainer failure onto a stable code.

    Empty patches, test failures, and ordinary model errors return None so
    the outer loop can keep treating them as algorithm results.
    """

    raw = text if isinstance(text, str) else str(text or "")
    if not raw:
        return None
    if _NOSPACE.search(raw):
        return INFRA_NOSPACE
    if _MEMORY.search(raw):
        return INFRA_MEMORY
    if _LDAP.search(raw):
        return INFRA_LDAP
    if _APPTAINER_START.search(raw):
        return INFRA_APPTAINER
    return None


def apply_infra_metadata(metadata: dict, code: str, message: str) -> dict:
    """Record an infrastructure stop without storing secret values."""

    metadata["infra_error_code"] = code
    metadata["evopilot_structural"] = True
    metadata["evopilot_failure_cluster"] = "infrastructure_error"
    metadata["error"] = redact_secrets(message)[:2000]
    return metadata


def stop_reason_for_infra(metadatas, *, continue_on_infra: bool) -> str | None:
    """Code of the first infrastructure failure, or None when the loop may continue."""

    if continue_on_infra:
        return None
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        code = metadata.get("infra_error_code")
        if code:
            return str(code)
    return None


def verify_worker_block(instance_id: str) -> dict | None:
    """Failed Verify result when another sandbox must not be started."""

    try:
        require_resources_from_env(phase="verify")
    except ResourceGateError as exc:
        return {
            "success": False,
            "instance_id": instance_id,
            "eval_result": "incomplete",
            "error": redact_secrets(str(exc)),
            "infra_error_code": exc.code,
        }
    return None


def raise_for_infra_results(results) -> None:
    for result in results:
        if not isinstance(result, dict):
            continue
        code = result.get("infra_error_code")
        if not code:
            continue
        message = result.get("error") or str(code)
        raise InfraEvalError(str(code), redact_secrets(str(message)))
