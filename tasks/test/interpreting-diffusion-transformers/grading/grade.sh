#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MD_PATH="${TASK_DIR}/task_description.md"

# Optional positional override for markdown path before flags.
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
  MD_PATH="$1"
  shift
fi

python3 "${SCRIPT_DIR}/grade.py" --md "${MD_PATH}" "$@"