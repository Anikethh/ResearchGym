from __future__ import annotations

import os
from pathlib import Path

from inspect_ai.tool import Tool, ToolError, tool


@tool(name="write_file")
def write_file_tool(timeout: int | None = None, user: str | None = None) -> Tool:
    """Write content to a file, creating parents as needed.

    Args:
        timeout: Unused placeholder for parity with other tools.
        user: Unused placeholder for parity with other tools.

    Parameters (when calling the tool):
        file_path (str, required): Path to write; can be relative to CODE_DIR or absolute.
        content (str, required): Full file contents to write. Overwrites any existing file.

    Returns:
        str: Status message noting whether the file was created or overwritten.

    Raises:
        ToolError: If the path is outside CODE_DIR or cannot be written.
    """

    async def execute(file_path: str, content: str) -> str:
        """Write a file inside the task workspace.

        Args:
            file_path: Path to write; may be relative to CODE_DIR or absolute but must stay within CODE_DIR.
            content: Full contents to write to the file. Overwrites if the file exists.

        Returns:
            Status message indicating whether the file was created or overwritten.

        Raises:
            ToolError: If the path is outside CODE_DIR or cannot be written.
        """
        # Resolve relative paths against the task workspace and confine to CODE_DIR.
        root = Path(os.environ.get("CODE_DIR", ".")).resolve()
        path = Path(file_path)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except Exception:
            raise ToolError(f"file_path must be within the workspace: {root}")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved.exists()
        resolved.write_text(content, encoding="utf-8")

        action = "overwrote" if existed else "created and wrote to new file"
        return f"Successfully {action}: {resolved}"

    return execute
