from __future__ import annotations

import os
from pathlib import Path

from inspect_ai.tool import Tool, ToolError, tool


@tool(name="replace")
def replace_tool(timeout: int | None = None, user: str | None = None) -> Tool:
    """Precise text replacement helper compatible with Gemini's replace tool.

    Args:
        timeout: Unused placeholder for parity with other tools.
        user: Unused placeholder for parity with other tools.

    Parameters (when calling the tool):
        file_path (str, required): Absolute path to the file to modify. Must live under CODE_DIR.
        old_string (str, required): Exact text to replace. Must uniquely match the file contents.
        new_string (str, required): Replacement text.
        expected_replacements (int, optional): Number of occurrences to replace (default: 1).

    Returns:
        str: Status message describing the edit that was performed.
    """

    async def execute(
        file_path: str, old_string: str, new_string: str, expected_replacements: int | None = None
    ) -> str:
        """Replace text in a file with strict matching.

        Args:
            file_path: Absolute path to the file to modify; must be inside CODE_DIR.
            old_string: Exact snippet to replace; include ample context so it is unique.
            new_string: Replacement snippet.
            expected_replacements: Number of occurrences to replace (default 1). Fails if count differs.

        Returns:
            Status string describing the change that was applied.

        Raises:
            ToolError: If the path is invalid, outside CODE_DIR, the file is missing,
                the match count is zero/non-unique, or creation rules are violated.
        """
        # Resolve and validate paths; only allow edits inside the workspace input tree.
        root = Path(os.environ.get("CODE_DIR", ".")).resolve()
        path = Path(file_path)
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except Exception:
            raise ToolError(f"file_path must be within the workspace: {root}")

        if not old_string:
            if resolved.exists():
                raise ToolError("Cannot create file: target already exists and old_string was empty.")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(new_string, encoding="utf-8")
            return f"Created new file: {resolved} with provided content."

        if not resolved.exists():
            raise ToolError(f"Failed to edit, file does not exist: {resolved}")

        content = resolved.read_text(encoding="utf-8")
        match_count = content.count(old_string)
        expected = 1 if expected_replacements is None else expected_replacements

        if match_count == 0:
            raise ToolError("Failed to edit, 0 occurrences found for the provided old_string.")
        if match_count != expected:
            raise ToolError(
                f"Failed to edit, expected {expected} occurrence(s) but found {match_count}."
            )

        new_content = content.replace(old_string, new_string, expected)
        resolved.write_text(new_content, encoding="utf-8")
        return f"Successfully modified file: {resolved} ({expected} replacement(s))."

    return execute
