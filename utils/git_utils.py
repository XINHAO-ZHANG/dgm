import os
import re
import git
import subprocess


def get_git_commit_hash(repo_path='.'):
    try:
        # Load the repository
        repo = git.Repo(repo_path)
        # Get the current commit hash
        commit_hash = repo.head.commit.hexsha
        return commit_hash
    except Exception as e:
        print("Error while getting git commit hash:", e)
        return None

def apply_patch(git_dname, patch_str):
    """
    Apply a patch to the repository at `git_dname`.
    """
    cmd = ["git", "-C", git_dname, "apply", "--reject", "-"]
    result = subprocess.run(
        cmd,
        input=patch_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    # Check if the patch was applied successfully
    if result.returncode != 0:
        print(f"apply_patch error: Patch did not fully apply. Return code: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
    else:
        print("apply_patch successful")

def diff_versus_commit(git_dname, commit):
    """
    Take a diff of `git_dname` current contents versus the `commit`, including untracked files,
    without modifying the repository state.
    """
    # Get diff of tracked files
    diff_cmd = ["git", "-C", git_dname, "diff", commit]
    result = subprocess.run(diff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    diff_output = result.stdout.decode()

    # Get list of untracked files
    untracked_files_cmd = ["git", "-C", git_dname, "ls-files", "--others", "--exclude-standard"]
    result = subprocess.run(untracked_files_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    untracked_files = result.stdout.decode().splitlines()

    # Generate diffs for untracked files
    for file in untracked_files:
        # Diff untracked file against /dev/null (empty file)
        file_path = os.path.join(git_dname, file)
        devnull = '/dev/null'
        if os.name == 'nt':  # Handle Windows
            devnull = 'NUL'
        diff_file_cmd = ["git", "-C", git_dname, "diff", "--no-index", devnull, file]
        result = subprocess.run(
            diff_file_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=git_dname,
            check=False
        )
        diff_file_output = result.stdout.decode('utf-8', errors='replace')
        diff_output += diff_file_output

    return diff_output

def reset_to_commit(git_dname, commit):
    """
    Reset the repository at `git_dname` to the given `commit`.
    """
    # Step 1: Hard-reset tracked files
    reset_cmd = ["git", "-C", git_dname, "reset", "--hard", commit]
    result_reset = subprocess.run(
        reset_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_reset.returncode != 0:
        print(f"reset_to_commit error: Failed to reset {git_dname} to commit '{commit}'. STDOUT: {result_reset.stdout} STDERR: {result_reset.stderr}")
    else:
        print(f"reset_to_commit successful: {commit}")

    # Step 2: Clean untracked files (the "new files") and directories
    clean_cmd = ["git", "-C", git_dname, "clean", "-fd"]
    result_clean = subprocess.run(
        clean_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_clean.returncode != 0:
        print(f"reset_to_commit clean error: Failed to clean {git_dname}. STDOUT: {result_clean.stdout} STDERR: {result_clean.stderr}")
    else:
        print(f"reset_to_commit clean successful: {commit}")


def filter_patch_by_files(patch_str, target_files):
    """
    Filters out the diff blocks related to any of the target_files in a patch string.

    Args:
        patch_str (str): The complete patch text.
        target_files (list[str]): A list of filenames for which to extract changes (e.g. ['affine_cipher.py', 'other.py']).

    Returns:
        str: A string containing only the diff blocks for the specified target files.
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False

    for line in lines:
        # When we encounter a new diff block header, check if the block is for any of the target files.
        if line.startswith("diff --git"):
            include_block = any(f"a/{target}" in line and f"b/{target}" in line for target in target_files)
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def remove_patch_by_files(patch_str, keyword='polyglot'):
    """
    Removes diff blocks related to files containing the keyword from a patch string.

    Args:
        patch_str (str): The complete patch text.
        keyword (str): Keyword to match in filenames for removal (default: 'polyglot').

    Returns:
        str: A string containing the patch with diff blocks for matching files removed.
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = True

    for line in lines:
        # When we encounter a new diff block header, check if the block contains the keyword
        if line.startswith("diff --git"):
            include_block = keyword.lower() not in line.lower()
        if include_block:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def ensure_trailing_newline(text: str) -> str:
    """Unified diffs must end with a newline or ``patch`` reports a mid-line EOF."""
    if not text:
        return text
    return text if text.endswith("\n") else text + "\n"


def _strip_newline(line: str) -> str:
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def _split_file_blocks(patch_str: str) -> list[list[str]]:
    lines = patch_str.splitlines(keepends=True)
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _hunk_counts(line: str) -> tuple[int, int] | None:
    match = _HUNK_HEADER.match(line)
    if not match:
        return None
    old = int(match.group(2) if match.group(2) is not None else 1)
    new = int(match.group(4) if match.group(4) is not None else 1)
    return old, new


def _block_hunks_complete(block_lines: list[str]) -> tuple[bool, str]:
    if not block_lines:
        return False, "empty_patch"
    joined = "".join(block_lines)
    if not joined.endswith("\n"):
        return False, "missing_trailing_newline"
    body = [_strip_newline(line) for line in block_lines]
    hunk_at = [index for index, line in enumerate(body) if line.startswith("@@")]
    if not hunk_at:
        last = body[-1]
        if last.startswith(("diff --git", "--- ", "+++ ", "new file mode", "deleted file mode")):
            return False, "truncated_before_hunk_body"
        return True, "ok"
    for position, start in enumerate(hunk_at):
        counts = _hunk_counts(body[start])
        if counts is None:
            return False, "malformed_hunk_header"
        old_need, new_need = counts
        end = hunk_at[position + 1] if position + 1 < len(hunk_at) else len(body)
        old_got = new_got = 0
        for line in body[start + 1 : end]:
            if not line:
                return False, "malformed_hunk_line"
            prefix = line[0]
            if prefix == "\\":
                continue
            if prefix == " ":
                old_got += 1
                new_got += 1
            elif prefix == "-":
                old_got += 1
            elif prefix == "+":
                new_got += 1
            else:
                return False, "malformed_hunk_line"
        if old_got != old_need or new_got != new_need:
            return False, "truncated_hunk"
    return True, "ok"


def patch_looks_well_formed(patch_str: str) -> tuple[bool, str]:
    """Cheap structural check before spending a Verify on a bad diff.

    Returns ``(ok, reason)``. A missing trailing newline and a hunk whose
    body is shorter than its ``@@`` header both count as malformed. Those
    are the failures ``patch`` reports as mid-line EOF or ``malformed patch``.
    """

    if not patch_str or not patch_str.strip():
        return False, "empty_patch"
    if "diff --git " not in patch_str:
        return False, "no_diff_git_header"
    if not patch_str.endswith("\n"):
        return False, "missing_trailing_newline"
    for block in _split_file_blocks(patch_str):
        ok, reason = _block_hunks_complete(block)
        if not ok:
            return False, reason
    return True, "ok"


def truncate_to_complete_file_diffs(patch_str: str) -> str:
    """Keep leading complete ``diff --git`` blocks and drop a truncated tail."""

    if not patch_str or "diff --git " not in patch_str:
        return ensure_trailing_newline(patch_str or "")
    kept: list[list[str]] = []
    for block in _split_file_blocks(patch_str):
        if not "".join(block).endswith("\n"):
            break
        ok, _reason = _block_hunks_complete(block)
        if not ok:
            break
        kept.append(block)
    return ensure_trailing_newline("".join("".join(block) for block in kept))


def prepare_model_patch(patch_content: str) -> tuple[str | None, dict]:
    """Repair a copied-out ``model_patch.diff`` before Verify.

    Returns ``(text, metadata)``. ``text`` is None when no complete file
    diff can be kept and Verify must not start.
    """

    notes: dict = {}
    fixed = ensure_trailing_newline(patch_content or "")
    ok, reason = patch_looks_well_formed(fixed)
    if ok:
        return fixed, notes
    salvaged = truncate_to_complete_file_diffs(fixed)
    ok_salvage, salvage_reason = patch_looks_well_formed(salvaged)
    if ok_salvage and salvaged.strip():
        notes["model_patch_salvaged"] = True
        notes["model_patch_malformed_reason"] = reason
        return salvaged, notes
    notes["model_patch_malformed_reason"] = reason
    notes["model_patch_salvage_failure"] = salvage_reason
    return None, notes


def preflight_model_patches(model_patch_paths) -> None:
    """Refuse Verify when a parent patch is still malformed.

    A missing trailing newline is written back. A truncated hunk is not
    applied across the task list.
    """

    if not model_patch_paths:
        return
    for model_patch_path in model_patch_paths:
        try:
            with open(model_patch_path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except OSError as exc:
            raise RuntimeError(f"Cannot read model patch {model_patch_path}: {exc}") from exc
        fixed = ensure_trailing_newline(content)
        ok, reason = patch_looks_well_formed(fixed)
        if not ok:
            raise RuntimeError(
                f"Refusing Verify: malformed model patch {model_patch_path} ({reason}). "
                "Re-run Improve or salvage with truncate_to_complete_file_diffs."
            )
        if fixed != content:
            with open(model_patch_path, "w", encoding="utf-8") as handle:
                handle.write(fixed)


def dry_run_apply_patch(git_dname: str, patch_str: str) -> tuple[bool, str]:
    """Return whether ``git apply --check`` accepts ``patch_str``."""

    patch_str = ensure_trailing_newline(patch_str)
    ok, reason = patch_looks_well_formed(patch_str)
    if not ok:
        return False, reason
    result = subprocess.run(
        ["git", "-C", git_dname, "apply", "--check", "-"],
        input=patch_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git_apply_check_failed").strip()
        return False, message
    return True, "ok"