#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOGS_DIR="${TASK_DIR}/logs"
MD_PATH="${TASK_DIR}/task_description.md"

# Optional positional overrides for logs directory and markdown path.
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  LOGS_DIR="$1"
  shift
fi
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  MD_PATH="$1"
  shift
fi

python3 "${SCRIPT_DIR}/grade.py" --logs "${LOGS_DIR}" --md "${MD_PATH}" "$@"
