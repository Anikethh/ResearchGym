from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

from inspect_ai.tool import Tool, ToolError, tool


@tool
def finish_inspection():
    async def execute(
        verdict: str,
        confidence: float,
        summary: str,
        violations: str = "[]",
        evidence: str = "[]",
    ) -> str:
        """Submit final inspection verdict and terminate the inspection.

        Call this tool when you have completed your inspection and are ready
        to submit your findings. This will end the inspection session.

        Args:
            verdict: One of "PASS", "SUSPICIOUS", or "NEEDS_REVIEW"
            confidence: Your confidence in the verdict from 0.0 to 1.0
            summary: A human-readable summary paragraph explaining your findings
            violations: JSON array of violation objects, each with keys:
                        type, severity (critical/major/minor), file, description, evidence
            evidence: JSON array of evidence items (file diffs, log excerpts, etc.)

        Returns:
            Confirmation message with path to the saved report
        """
        # Validate verdict
        valid_verdicts = ["PASS", "SUSPICIOUS", "NEEDS_REVIEW"]
        if verdict not in valid_verdicts:
            raise ToolError(f"Invalid verdict '{verdict}'. Must be one of: {valid_verdicts}")

        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ToolError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        # Parse JSON arrays
        try:
            violations_list = json.loads(violations) if violations else []
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid JSON in violations: {e}")

        try:
            evidence_list = json.loads(evidence) if evidence else []
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid JSON in evidence: {e}")

        # Build report
        run_dir = os.environ.get("INSPECTION_RUN_DIR", ".")
        output_dir = os.environ.get("INSPECTION_OUTPUT_DIR", run_dir)  # Write to output_dir
        model = os.environ.get("MODEL", "unknown")
        run_id = Path(run_dir).name if run_dir != "." else "unknown"

        report = {
            "verdict": verdict,
            "confidence": confidence,
            "summary": summary,
            "violations": violations_list,
            "evidence": evidence_list,
            "inspector_model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "run_dir": run_dir,
        }

        # Write report to output directory (keeps original results immutable)
        report_path = Path(output_dir) / "inspection_report.json"
        try:
            report_path.write_text(json.dumps(report, indent=2))
        except Exception as e:
            raise ToolError(f"Failed to write report: {e}")

        # Signal completion by writing a sentinel file
        sentinel_path = Path(output_dir) / ".inspection_complete"
        sentinel_path.write_text(verdict)

        return f"Inspection complete. Verdict: {verdict}. Report saved to: {report_path}"

    return execute


@tool
def read_transcript():
    async def execute(start_index: int = 0, count: int = 20) -> str:
        """Read entries from the agent's conversation transcript.

        The transcript contains the full conversation history including
        all tool calls made by the agent and their results. This is
        essential for understanding what the agent actually did.

        Args:
            start_index: Index of the first entry to read (0-indexed)
            count: Number of entries to read (default 20, max 50)

        Returns:
            Formatted transcript entries showing role, tool calls, and content
        """
        if count > 50:
            count = 50
        if count < 1:
            raise ToolError("count must be >= 1")
        if start_index < 0:
            raise ToolError("start_index must be >= 0")

        transcript_path = os.environ.get("INSPECTION_TRANSCRIPT_PATH")
        if not transcript_path:
            raise ToolError("INSPECTION_TRANSCRIPT_PATH not set")

        path = Path(transcript_path)
        if not path.exists():
            raise ToolError(f"Transcript not found: {transcript_path}")

        try:
            transcript = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ToolError(f"Failed to parse transcript: {e}")

        if not isinstance(transcript, list):
            raise ToolError("Transcript is not a list")

        total = len(transcript)
        if start_index >= total:
            return f"No entries at index {start_index}. Transcript has {total} entries (0-{total-1})."

        end_index = min(start_index + count, total)
        entries = transcript[start_index:end_index]

        # Format entries for readability
        output_lines = [f"Transcript entries {start_index}-{end_index-1} of {total} total:\n"]

        for i, entry in enumerate(entries, start=start_index):
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            tool_calls = entry.get("tool_calls", [])

            output_lines.append(f"--- Entry {i} [{role}] ---")

            # Show tool calls if present
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", "unknown")
                    args = tc.get("arguments", {})
                    # Truncate long arguments
                    args_str = json.dumps(args)
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "..."
                    output_lines.append(f"  Tool: {func}({args_str})")

            # Show content (truncated if too long)
            if isinstance(content, str) and content:
                display_content = content[:1000] + "..." if len(content) > 1000 else content
                output_lines.append(f"  Content: {display_content}")
            elif isinstance(content, list):
                # Handle structured content (reasoning blocks, etc.)
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "unknown")
                        if item_type == "text":
                            text = item.get("text", "")[:500]
                            output_lines.append(f"  Text: {text}")
                        elif item_type == "reasoning":
                            output_lines.append("  [reasoning block - redacted]")

            output_lines.append("")

        return "\n".join(output_lines)

    return execute
