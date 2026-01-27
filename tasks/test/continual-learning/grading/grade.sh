#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (relative to task root)
LOGS_DIR="${TASK_DIR}/logs"
MD_PATH="${TASK_DIR}/task_description.md"

# Usage for agents:
#   ./grading/grade.sh                # grade every table using default logs/md locations
#   ./grading/grade.sh logs/custom    # override logs directory
#   ./grading/grade.sh logs custom_md --table imagenet_r --table cub200
# Flags after the optional positional overrides are forwarded to grade.py.

# Positional overrides for logs and md
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  LOGS_DIR="$1"
  shift
fi
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  MD_PATH="$1"
  shift
fi

python3 "${SCRIPT_DIR}/grade.py" --logs "${LOGS_DIR}" --md "${MD_PATH}" "$@"

