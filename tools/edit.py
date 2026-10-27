from pathlib import Path
import subprocess
from typing import Optional, Sequence


def tool_info():
    return {
        "name": "editor",
        "description": """Custom editing tool for viewing, creating, and editing files\n
* State is persistent across command calls and discussions with the user.\n
* If `path` is a file, `view` displays the file with line numbers. Optional `view_range` [start_line, end_line] (1-indexed, inclusive; end_line -1 means EOF) shows only that slice. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep (`view_range` is ignored).\n
* The `create` command cannot be used if the specified `path` already exists as a file.\n
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`.\n
* The `edit` command overwrites the entire file with the provided `file_text`.\n
* Partial/line-range edits are not supported; use `edit` with the full file text.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "edit"],
                    "description": "The command to run: `view`, `create`, or `edit`."
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                    "type": "string"
                },
                "file_text": {
                    "description": "Required parameter of `create` or `edit` command, containing the content for the entire file.",
                    "type": "string"
                },
                "view_range": {
                    "description": "Optional for `view` on a file. [start_line, end_line], 1-indexed inclusive. Use end_line -1 for EOF. Ignored for directories.",
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["command", "path"]
        }
    }


def maybe_truncate(content: str, max_length: int = 10000) -> str:
    """Truncate long content and add marker."""
    if len(content) > max_length:
        return content[:max_length] + "\n<response clipped>"
    return content


def validate_path(path: str, command: str) -> Path:
    """
    Validate the file path for each command:
      - 'view': path may be a file or directory; must exist.
      - 'create': path must not exist (for new file creation).
      - 'edit': path must exist (for overwriting).
    """
    path_obj = Path(path)

    # Check if it's an absolute path
    if not path_obj.is_absolute():
        raise ValueError(
            f"The path {path} is not an absolute path (must start with '/')."
        )

    if command == "view":
        # Path must exist
        if not path_obj.exists():
            raise ValueError(f"The path {path} does not exist.")
    elif command == "create":
        # Path must not exist
        if path_obj.exists():
            raise ValueError(f"Cannot create new file; {path} already exists.")
    elif command == "edit":
        # Path must exist and must be a file
        if not path_obj.exists():
            raise ValueError(f"The file {path} does not exist.")
        if path_obj.is_dir():
            raise ValueError(f"{path} is a directory and cannot be edited as a file.")
    else:
        raise ValueError(f"Unknown or unsupported command: {command}")

    return path_obj


def format_output(content: str, path: str, init_line: int = 1) -> str:
    """Format output with line numbers (for file content)."""
    content = maybe_truncate(content)
    content = content.expandtabs()
    numbered_lines = [
        f"{i + init_line:6}\t{line}"
        for i, line in enumerate(content.split("\n"))
    ]
    return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered_lines) + "\n"


def read_file(path: Path) -> str:
    """Read and return the entire file contents."""
    try:
        return path.read_text()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")


def write_file(path: Path, content: str):
    """Write (overwrite) entire file contents."""
    try:
        path.write_text(content)
    except Exception as e:
        raise ValueError(f"Failed to write file: {e}")


def _normalize_view_range(view_range: Optional[Sequence[int]], n_lines: int) -> tuple[int, int]:
    """Return 0-based [start, end) slice bounds from a 1-indexed inclusive range."""
    if view_range is None:
        return 0, n_lines
    if not isinstance(view_range, (list, tuple)) or len(view_range) != 2:
        raise ValueError("view_range must be [start_line, end_line]")
    start_line, end_line = int(view_range[0]), int(view_range[1])
    if start_line < 1:
        raise ValueError(f"view_range start_line must be >= 1, got {start_line}")
    start_idx = start_line - 1
    if start_idx >= n_lines:
        raise ValueError(
            f"view_range start_line {start_line} is past end of file ({n_lines} lines)"
        )
    if end_line == -1:
        end_idx = n_lines
    else:
        if end_line < start_line:
            raise ValueError(
                f"view_range end_line {end_line} must be >= start_line {start_line} or -1"
            )
        end_idx = min(n_lines, end_line)  # inclusive end -> exclusive slice
    return start_idx, end_idx


def view_path(path_obj: Path, view_range: Optional[Sequence[int]] = None) -> str:
    """View file contents (optionally a line range) or directory listing."""
    if path_obj.is_dir():
        # For directories: list non-hidden files up to 2 levels deep
        try:
            result = subprocess.run(
                ['find', str(path_obj), '-maxdepth', '2', '-not', '-path', '*/\\.*'],
                capture_output=True,
                text=True
            )
            if result.stderr:
                return f"Error listing directory: {result.stderr}"
            return (
                f"Here's the files and directories up to 2 levels deep in {path_obj}, excluding hidden items:\n"
                + result.stdout
            )
        except Exception as e:
            raise ValueError(f"Failed to list directory: {e}")

    # If it's a file, show (a slice of) the file with line numbers
    content = read_file(path_obj)
    lines = content.split("\n")
    start_idx, end_idx = _normalize_view_range(view_range, len(lines))
    selected = "\n".join(lines[start_idx:end_idx])
    return format_output(selected, str(path_obj), init_line=start_idx + 1)


def tool_function(
    command: str,
    path: str,
    file_text: str = None,
    view_range: Optional[Sequence[int]] = None,
) -> str:
    """
    Main tool function that handles:
      - 'view'  : View the entire file (or view_range slice) or directory listing
      - 'create': Create a new file with the given file_text
      - 'edit'  : Overwrite an existing file with file_text
    """
    try:
        path_obj = validate_path(path, command)

        if command == "view":
            return view_path(path_obj, view_range=view_range)

        elif command == "create":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'create' command.")
            write_file(path_obj, file_text)
            return f"File created successfully at: {path}"

        elif command == "edit":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'edit' command.")
            write_file(path_obj, file_text)
            return f"File at {path} has been overwritten with new content."

        else:
            raise ValueError(f"Unknown command: {command}")

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage (absolute path required by validate_path)
    sample = Path(__file__).resolve().parent.parent / "coding_agent.py"
    result = tool_function("view", str(sample), view_range=[1, 10])
    print(result)
