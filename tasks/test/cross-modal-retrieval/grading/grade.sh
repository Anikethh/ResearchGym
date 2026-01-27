#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_ROOT="${TASK_DIR}/output"
DEFAULT_MD="${TASK_DIR}/task_description.md"

ROOT="${DEFAULT_ROOT}"
MD="${DEFAULT_MD}"

if [[ $# -ge 1 && "${1#-}" == "$1" ]]; then
  ROOT="$1"
  shift
fi
if [[ $# -ge 1 && "${1#-}" == "$1" ]]; then
  MD="$1"
  shift
fi

python3 "${SCRIPT_DIR}/grade.py" --task-dir "${TASK_DIR}" --root "${ROOT}" --md "${MD}" "$@"
