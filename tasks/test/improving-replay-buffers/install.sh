#!/usr/bin/env bash
set -euo pipefail

if [[ "${SKIP_TASK_INSTALL:-0}" == "1" ]]; then
    echo "[install.sh] SKIP_TASK_INSTALL=1; skipping task install."
    exit 0
fi

REQ_FILE="/task/requirements.txt"
PY_BIN="${RL_PYTHON:-}"

if [[ -z "${PY_BIN}" || ! -x "${PY_BIN}" ]]; then
    if command -v python3.8 >/dev/null 2>&1; then
        PY_BIN="$(command -v python3.8)"
    elif command -v rlpython >/dev/null 2>&1; then
        PY_BIN="$(command -v rlpython)"
    else
        echo "[install.sh] Could not locate an RL-compatible Python interpreter." >&2
        exit 1
    fi
fi

echo "[install.sh] Using interpreter: ${PY_BIN}"

TMP_REQ="$(mktemp)"
grep -v "dm-control" "${REQ_FILE}" > "${TMP_REQ}"
${PY_BIN} -m pip install -r "${TMP_REQ}"

# Optionally mirror installs into an additional interpreter (e.g., system python3.10)
if [[ -z "${SECONDARY_PY_BIN:-}" && -x "/opt/py310/bin/python" ]]; then
    SECONDARY_PY_BIN="/opt/py310/bin/python"
fi
if [[ -n "${SECONDARY_PY_BIN:-}" && -x "${SECONDARY_PY_BIN}" ]]; then
    echo "[install.sh] Also installing requirements into ${SECONDARY_PY_BIN}"
    "${SECONDARY_PY_BIN}" -m pip install -r "${TMP_REQ}"
fi
rm -f "${TMP_REQ}"

# Manual install for dm-control with duplicate typedef guard fix
WORKDIR="$(mktemp -d)"
echo "[install.sh] Cloning dm_control..."
git clone https://github.com/deepmind/dm_control.git "${WORKDIR}/dm_control"
pushd "${WORKDIR}/dm_control" >/dev/null
git checkout f4e5e2336991017361f563db7e676fd08857076f >/dev/null

echo "[install.sh] Applying duplicate typedef guard patch..."
python - <<'PY'
from pathlib import Path

path = Path("dm_control/autowrap/binding_generator.py")
text = path.read_text()
marker = "_researchgym_binding_patch"
needle = "self.typedefs_dict.update({token.name: token.typename})"
if marker not in text and needle in text:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            indent = line[: line.index("self")].replace("\t", "    ")
            replacement = [
                f"{indent}existing = self.typedefs_dict.get(token.name)",
                f"{indent}if existing is None:",
                f"{indent}    self.typedefs_dict[token.name] = token.typename",
                f"{indent}elif existing != token.typename:",
                f'{indent}    if existing == "double" and token.typename == "float":',
                f"{indent}        pass  # {marker}",
                f"{indent}    else:",
                f"{indent}        raise ValueError("
                f'\"Conflicting typedef for {{}}: {{}} vs {{}}\".format(token.name, existing, token.typename)'
                f")",
            ]
            lines[i : i + 1] = replacement
            break
    path.write_text("\n".join(lines) + "\n")
PY

echo "[install.sh] Installing dm_control from source..."
${PY_BIN} -m pip install --no-deps .
popd >/dev/null
rm -rf "${WORKDIR}"

# Trigger mujoco-py compilation early so failures surface during setup
${PY_BIN} - <<'PY'
import mujoco_py  # noqa: F401
print("[install.sh] mujoco_py import succeeded.")
PY
