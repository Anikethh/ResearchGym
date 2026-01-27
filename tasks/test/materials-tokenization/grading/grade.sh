#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Convenience wrapper around grade.py so agents can stay in bash-land.
# Examples:
#   ./grading/grade.sh                             # run every subtask, update both tables
#   ./grading/grade.sh --table generation          # only refresh the generation table
#   METHOD_NAME=MyRun ./grading/grade.sh --subtask rc --table classification
#   ./grading/grade.sh --aggregate-only --no-markdown

python3 "${SCRIPT_DIR}/grade.py" --task-dir "${TASK_DIR}" "$@"
