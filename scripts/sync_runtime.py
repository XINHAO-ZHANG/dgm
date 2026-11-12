"""Copy EvoPilot's agent, grading, and sandbox runtime into this checkout.

The DGM baseline is compared with EvoPilot, so everything both of them run must
be the same code: the coding agent that starts the evolution, the harness that
grades it, the task images it is graded in, the packages installed in every
sandbox, and what the self-editing agent sees of its own repository (Dockerfile
and .dockerignore). EvoPilot is where those files are maintained. This script
copies them from an EvoPilot checkout in which they are committed, and writes
EVOPILOT_RUNTIME.json with that EvoPilot commit and a hash per file.

``--check`` compares this checkout with the record and lists what the
self-improve image would contain in each repository. DGM_outer.py refuses an
aligned run (``--split``) while a runtime file differs from the record.

DGM keeps its own method: DGM_outer.py, self_improve_step.py, and
prompts/self_improvement_prompt.py (the upstream diagnosis prompt) are not copied.

    python scripts/sync_runtime.py [--source ../EvoPilot]
    python scripts/sync_runtime.py --check

``--allow_uncommitted`` syncs anyway and lists the uncommitted files in the
record; such a record fails the check until a sync from committed files.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = "EVOPILOT_RUNTIME.json"
RUNTIME_FILES = (
    # The coding agent: it starts the evolution and solves every task.
    "coding_agent_polyglot.py",
    "llm.py",
    "llm_withtools.py",
    "tools/__init__.py",
    "tools/bash.py",
    "tools/edit.py",
    "prompts/tooluse_prompt.py",
    "prompts/testrepo_prompt.py",
    # Installed in the self-improve image and in every task sandbox.
    "requirements.txt",
    # Grading and task images.
    "polyglot/harness.py",
    "polyglot/constants.py",
    "polyglot/dockerfiles.py",
    "polyglot/docker_build.py",
    "polyglot/docker_utils.py",
    "polyglot/test_spec.py",
    "polyglot/run_evaluation.py",
    # Sandbox runtime.
    "utils/__init__.py",
    "utils/apptainer_compat.py",
    "utils/apptainer_errors.py",
    "utils/container_runtime.py",
    "utils/docker_utils.py",
    "utils/host_resources.py",
    "utils/secret_redaction.py",
    "utils/fs_copy.py",
    "utils/git_utils.py",
    "utils/common_utils.py",
    # The self-improve image and what the self-editing agent sees of the repository.
    "Dockerfile",
    ".dockerignore",
)
# Not tracked by git in either repository; compared, never copied.
DATA_FILES = ("polyglot/polyglot_benchmark_metadata.json",)
# The self-improve container removes these before the agent runs (self_improve_step.py).
REMOVED_IN_CONTAINER = ("coding_agent.py", "utils/eval_utils.py", "utils/swe_log_parsers.py")


def sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def git(repo: str, *args: str) -> str:
    # rstrip only: a porcelain status line starts with a space for an unstaged change.
    return subprocess.check_output(["git", "-C", repo, *args], text=True).rstrip()


def sync(source: str, root: str = ROOT, *, allow_uncommitted: bool = False) -> dict:
    """Copy RUNTIME_FILES from ``source`` and write the record. Refuses uncommitted sources."""

    missing = [path for path in RUNTIME_FILES if not os.path.isfile(os.path.join(source, path))]
    if missing:
        raise SystemExit(f"{source} lacks {', '.join(missing)}")
    dirty = git(source, "status", "--porcelain", "--", *RUNTIME_FILES)
    if dirty and not allow_uncommitted:
        raise SystemExit(f"Commit these in {source} first; the record names a commit:\n{dirty}")
    for path in RUNTIME_FILES:
        target = os.path.join(root, path)
        os.makedirs(os.path.dirname(target) or root, exist_ok=True)
        shutil.copyfile(os.path.join(source, path), target)
    record = {
        "source": "EvoPilot",
        "source_commit": git(source, "rev-parse", "HEAD"),
        "synced_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "files": {path: sha256(os.path.join(root, path)) for path in RUNTIME_FILES},
        "data": {path: sha256(os.path.join(source, path)) for path in DATA_FILES},
        "source_uncommitted": [line[3:] for line in dirty.splitlines()],
    }
    with open(os.path.join(root, RECORD), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record


def runtime_mismatches(root: str = ROOT) -> list[str]:
    """Runtime files that differ from EVOPILOT_RUNTIME.json; the record itself when missing."""

    path = os.path.join(root, RECORD)
    if not os.path.isfile(path):
        return [f"{RECORD} (missing: run scripts/sync_runtime.py)"]
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    mismatches = [
        f"{name} (synced from an uncommitted EvoPilot file; commit it there and sync again)"
        for name in record.get("source_uncommitted") or ()
    ]
    for name, digest in (record.get("files") or {}).items():
        if sha256(os.path.join(root, name)) != digest:
            mismatches.append(name)
    for name, digest in (record.get("data") or {}).items():
        actual = sha256(os.path.join(root, name))
        if digest and actual != digest:
            mismatches.append(f"{name} (dataset differs from the one EvoPilot uses)")
    return mismatches


def context_files(repo: str) -> set[str]:
    """Files the self-improve image gets from ``repo``, as the agent sees them after container setup."""

    sys.path.insert(0, ROOT)
    from utils.apptainer_compat import _dockerignore_patterns, _path_ignored

    patterns = _dockerignore_patterns(__import__("pathlib").Path(repo))
    files = set()
    for current, dirs, names in os.walk(repo):
        rel_dir = os.path.relpath(current, repo)
        rel_dir = "" if rel_dir == "." else rel_dir
        dirs[:] = [d for d in dirs if d != ".git" and not _path_ignored(os.path.join(rel_dir, d), patterns, is_dir=True)]
        for name in names:
            rel = os.path.join(rel_dir, name)
            if not _path_ignored(rel, patterns, is_dir=False):
                files.add(rel)
    files -= set(REMOVED_IN_CONTAINER)
    if "coding_agent_polyglot.py" in files:
        files.discard("coding_agent_polyglot.py")
        files.add("coding_agent.py")
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default=os.path.join(os.path.dirname(ROOT), "EvoPilot"))
    parser.add_argument("--check", action="store_true", help="Compare with the record; copy nothing.")
    parser.add_argument("--allow_uncommitted", action="store_true", help="Development only: sync uncommitted files and say so in the record.")
    args = parser.parse_args(argv)
    if not args.check:
        record = sync(args.source, allow_uncommitted=args.allow_uncommitted)
        print(f"Synced {len(record['files'])} files from EvoPilot {record['source_commit'][:7]}")
    mismatches = runtime_mismatches()
    for name in mismatches:
        print(f"differs from {RECORD}: {name}")
    if os.path.isdir(args.source):
        ours, theirs = context_files(ROOT), context_files(args.source)
        for name in sorted(ours - theirs):
            print(f"self-improve image: only in dgm: {name}")
        for name in sorted(theirs - ours):
            print(f"self-improve image: only in EvoPilot: {name}")
        if ours == theirs:
            print(f"self-improve image: same {len(ours)} files in both repositories")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
