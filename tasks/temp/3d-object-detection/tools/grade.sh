# ============================
# Usage
# cd "/ResearchGym/original_tasks/test/3d-object-detection"
# bash tools/grade.sh \
#   --scannet-config projects_sparse/configs/two_stage/tr3d_scannet-3d-18class_cpdet3d_2.py \
#   --scannet-ckpt /abs/path/to/scannet_checkpoint.pth \
#   --sunrgbd-config projects_sparse/configs/two_stage/tr3d_sunrgbd-3d-10class_cpdet3d_2.py \
#   --sunrgbd-ckpt /abs/path/to/sunrgbd_checkpoint.pth \
#   --gpus 1

    # --kitti-config /abs/path/to/kitti_config.py \
#   --kitti-ckpt /abs/path/to/kitti_checkpoint.pth
# ============================

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
ARTIFACTS_DIR="${TASK_DIR}/artifacts"
OUTPUT_MD="${TASK_DIR}/task_description.md"
GPUS=1

# Optional explicit paths (override discovery)
SCAN_CFG=""
SCAN_CKPT=""
SUN_CFG=""
SUN_CKPT=""
KITTI_CFG=""
KITTI_CKPT=""

usage() {
  cat <<USAGE
Usage: bash tools/grade.sh [options]

Options:
  --artifacts-dir PATH          Root dir where LLM saved artifacts (default: tasks/test/3d-object-detection/artifacts)
                                Expected subdirs: scannet/, sunrgbd/, kitti/ each with a config .py and a .pth checkpoint.
  --scannet-config PATH         Override ScanNet config path (.py)
  --scannet-ckpt PATH           Override ScanNet checkpoint path (.pth)
  --sunrgbd-config PATH         Override SUN RGB-D config path (.py)
  --sunrgbd-ckpt PATH           Override SUN RGB-D checkpoint path (.pth)
  --kitti-config PATH           Override KITTI config path (.py)
  --kitti-ckpt PATH             Override KITTI checkpoint path (.pth)
  --gpus N                      Number of GPUs for distributed eval (default: 1)
  --output-md PATH              Markdown to update (default: tasks/test/3d-object-detection/task_description.md)
  -h | --help                   Show this help

Notes:
  - Indoor metrics (ScanNet V2, SUN RGB-D) are reported as percentages (x100 of mAP_0.25/mAP_0.50).
  - KITTI metrics use AP R40 as returned by evaluator (already in %).
  - Only datasets with both config and checkpoint will be evaluated.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts-dir) ARTIFACTS_DIR="$2"; shift 2;;
    --scannet-config) SCAN_CFG="$2"; shift 2;;
    --scannet-ckpt) SCAN_CKPT="$2"; shift 2;;
    --sunrgbd-config) SUN_CFG="$2"; shift 2;;
    --sunrgbd-ckpt) SUN_CKPT="$2"; shift 2;;
    --kitti-config) KITTI_CFG="$2"; shift 2;;
    --kitti-ckpt) KITTI_CKPT="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --output-md) OUTPUT_MD="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

ensure_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    return 1
  fi
  return 0
}

discover_artifact_pair() {
  # $1 = dataset key (scannet|sunrgbd|kitti)
  # stdout: CONFIG\nCKPT (or nothing if not found)
  local dset="$1"
  local base_dir="${ARTIFACTS_DIR}/${dset}"
  if [[ -d "$base_dir" ]]; then
    local cfg=""
    local ckpt=""
    # Prefer explicitly named config.py, else first .py
    if [[ -f "${base_dir}/config.py" ]]; then
      cfg="${base_dir}/config.py"
    else
      cfg="$(ls -1 "${base_dir}"/*.py 2>/dev/null | head -n1 || true)"
    fi
    # Prefer best*.pth, then latest*.pth, then any .pth
    ckpt="$(ls -1 "${base_dir}"/best*.pth 2>/dev/null | head -n1 || true)"
    if [[ -z "$ckpt" ]]; then
      ckpt="$(ls -1 "${base_dir}"/latest*.pth 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "$ckpt" ]]; then
      ckpt="$(ls -1 "${base_dir}"/*.pth 2>/dev/null | head -n1 || true)"
    fi

    if [[ -n "$cfg" && -n "$ckpt" ]]; then
      echo "$cfg"
      echo "$ckpt"
      return 0
    fi
  fi
  return 1
}

LOG_DIR="${TASK_DIR}/work_dirs/grade_logs"
mkdir -p "$LOG_DIR"

PYTHONPATH_PREFIX="$(cd "${SCRIPT_DIR}/.." && pwd)"

run_eval_single_gpu() {
  local config="$1"
  local ckpt="$2"
  local log_file="$3"
  PYTHONPATH="${PYTHONPATH_PREFIX}:$PYTHONPATH" \
  python "${SCRIPT_DIR}/test.py" "$config" "$ckpt" --eval mAP | tee "$log_file"
}

run_eval_distributed() {
  local config="$1"
  local ckpt="$2"
  local gpus="$3"
  local log_file="$4"
  bash "${SCRIPT_DIR}/dist_test.sh" "$config" "$ckpt" "$gpus" | tee "$log_file"
}

parse_indoor_metrics() {
  # $1 = log file; prints: map25 map50 mar25 mar50 (as percentages, 1 decimal)
  local log_file="$1"
  python - "$log_file" <<'PY'
import sys, re, ast
from math import isnan

text = open(sys.argv[1], 'r', errors='ignore').read()
# Find last dict-like block in the log
candidates = re.findall(r'\{[\s\S]*?\}', text)
metrics = {}
for s in reversed(candidates):
    try:
        d = ast.literal_eval(s)
    except Exception:
        continue
    if isinstance(d, dict) and d:
        metrics = d
        break

def fmt_pct(x):
    try:
        v = float(x) * 100.0
        if isnan(v):
            return ''
        return f"{v:.1f}"
    except Exception:
        return ''

map25 = fmt_pct(metrics.get('mAP_0.25', ''))
map50 = fmt_pct(metrics.get('mAP_0.50', ''))
mar25 = fmt_pct(metrics.get('mAR_0.25', ''))
mar50 = fmt_pct(metrics.get('mAR_0.50', ''))
print("\n".join([map25, map50, mar25, mar50]))
PY
}

parse_kitti_metrics() {
  # $1 = log file; prints: car_3d_e car_3d_m car_3d_h car_bev_e car_bev_m car_bev_h (percentages as is, 1 decimal)
  local log_file="$1"
  python - "$log_file" <<'PY'
import sys, re, ast

text = open(sys.argv[1], 'r', errors='ignore').read()
candidates = re.findall(r'\{[\s\S]*?\}', text)
metrics = {}
for s in reversed(candidates):
    try:
        d = ast.literal_eval(s)
    except Exception:
        continue
    if isinstance(d, dict) and d:
        metrics = d
        break

def pick(key_strict, key_loose):
    if key_strict in metrics:
        return metrics[key_strict]
    if key_loose in metrics:
        return metrics[key_loose]
    return ''

def fmt(x):
    try:
        v = float(x)
        return f"{v:.1f}"
    except Exception:
        return ''

car_3d_e = fmt(pick('KITTI/Car_3D_AP40_easy_strict', 'KITTI/Car_3D_AP40_easy_loose'))
car_3d_m = fmt(pick('KITTI/Car_3D_AP40_moderate_strict', 'KITTI/Car_3D_AP40_moderate_loose'))
car_3d_h = fmt(pick('KITTI/Car_3D_AP40_hard_strict', 'KITTI/Car_3D_AP40_hard_loose'))
car_bev_e = fmt(pick('KITTI/Car_BEV_AP40_easy_strict', 'KITTI/Car_BEV_AP40_easy_loose'))
car_bev_m = fmt(pick('KITTI/Car_BEV_AP40_moderate_strict', 'KITTI/Car_BEV_AP40_moderate_loose'))
car_bev_h = fmt(pick('KITTI/Car_BEV_AP40_hard_strict', 'KITTI/Car_BEV_AP40_hard_loose'))
print("\n".join([car_3d_e, car_3d_m, car_3d_h, car_bev_e, car_bev_m, car_bev_h]))
PY
}

update_markdown() {
  # Inputs via env vars:
  #   SCAN_MAP25, SCAN_MAP50
  #   SUN_MAP25, SUN_MAP50
  #   KITTI_3D_E, KITTI_3D_M, KITTI_3D_H, KITTI_BEV_E, KITTI_BEV_M, KITTI_BEV_H
  # $1 = markdown path
  local md="$1"
  python - "$md" <<'PY'
import os, sys, re
md_path = sys.argv[1]
with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

def replace_row(after_header_regex, your_row_regex, new_cells_func):
    # Find section start by header regex, then find Your Method row to replace
    start_idx = None
    for i, ln in enumerate(lines):
        if re.search(after_header_regex, ln):
            start_idx = i
            break
    if start_idx is None:
        return
    for j in range(start_idx, len(lines)):
        if re.match(your_row_regex, lines[j].strip()):
            raw = lines[j]
            cells = [c.strip() for c in raw.split('|')]
            # Ensure leading and trailing '|'
            # cells indices: 0 .. N .. last (first and last are empty due to split around starting/ending '|')
            updated = new_cells_func(cells)
            # If existing row already contains digits, append a new row; else replace placeholder
            if re.search(r'\d', raw):
                lines.insert(j + 1, '|'.join(updated))
            else:
                lines[j] = '|'.join(updated)
            break

# Table 1: Indoor
SCAN_MAP25 = os.environ.get('SCAN_MAP25', '')
SCAN_MAP50 = os.environ.get('SCAN_MAP50', '')
SUN_MAP25  = os.environ.get('SUN_MAP25', '')
SUN_MAP50  = os.environ.get('SUN_MAP50', '')

def upd_indoor(cells):
    # Expected columns: | Methods | Present at | Paradigm | ScanNet V2 mAP@0.25 | ScanNet V2 mAP@0.5 | SUN RGB-D mAP@0.25 | SUN RGB-D mAP@0.5 |
    # cells example: ['', ' Your Method ', ' -- ', ' -- ', ' -- ', ' -- ', ' -- ', ' -- ', ''] -> length 9
    # Keep first 3 intact; set last 4 if available
    if len(cells) >= 9:
        cells[2] = ' – '
        cells[3] = ' – '
        cells[4] = f' {SCAN_MAP25 or "--"} '
        cells[5] = f' {SCAN_MAP50 or "--"} '
        cells[6] = f' {SUN_MAP25 or "--"} '
        cells[7] = f' {SUN_MAP50 or "--"} '
    return cells

replace_row(r'^For indoor setting Table 1', r'^\|\s*Your Method\s*\|', upd_indoor)

# Table 2: KITTI
K3E = os.environ.get('KITTI_3D_E', '')
K3M = os.environ.get('KITTI_3D_M', '')
K3H = os.environ.get('KITTI_3D_H', '')
KBE = os.environ.get('KITTI_BEV_E', '')
KBM = os.environ.get('KITTI_BEV_M', '')
KBH = os.environ.get('KITTI_BEV_H', '')

def upd_kitti(cells):
    # Expected columns: | Methods | Present at | Paradigm | Car-3D AP (R40) Easy | Mod. | Hard | Car-BEV AP (R40) Easy | Mod. | Hard |
    if len(cells) >= 11:
        cells[2] = ' – '
        cells[3] = ' – '
        cells[4] = f' {K3E or "--"} '
        cells[5] = f' {K3M or "--"} '
        cells[6] = f' {K3H or "--"} '
        cells[7] = f' {KBE or "--"} '
        cells[8] = f' {KBM or "--"} '
        cells[9] = f' {KBH or "--"} '
    return cells

replace_row(r'^For outdoor scenes Table 2', r'^\|\s*Your Method\s*\|', upd_kitti)

# Table 3: Semi-supervised (ScanNet V2)
def upd_semi(cells):
    # Expected columns: | Methods | Present at | mAP@0.25 | mAP@0.5 |
    if len(cells) >= 6:
        # Preserve Present at (index 2); fill indices 3 and 4
        cells[3] = f' {SCAN_MAP25 or "--"} '
        cells[4] = f' {SCAN_MAP50 or "--"} '
    return cells

replace_row(r'^Table 3\.', r'^\|\s*Your Method\s*\|', upd_semi)

with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
PY
}

#
# 1) Resolve artifacts for each dataset
#
if [[ -z "$SCAN_CFG" || -z "$SCAN_CKPT" ]]; then
  if pair=$(discover_artifact_pair scannet); then
    SCAN_CFG=$(echo "$pair" | sed -n '1p')
    SCAN_CKPT=$(echo "$pair" | sed -n '2p')
  fi
fi
if [[ -z "$SUN_CFG" || -z "$SUN_CKPT" ]]; then
  if pair=$(discover_artifact_pair sunrgbd); then
    SUN_CFG=$(echo "$pair" | sed -n '1p')
    SUN_CKPT=$(echo "$pair" | sed -n '2p')
  fi
fi
if [[ -z "$KITTI_CFG" || -z "$KITTI_CKPT" ]]; then
  if pair=$(discover_artifact_pair kitti); then
    KITTI_CFG=$(echo "$pair" | sed -n '1p')
    KITTI_CKPT=$(echo "$pair" | sed -n '2p')
  fi
fi

#
# 2) Evaluate where possible
#
SCAN_MAP25=""; SCAN_MAP50=""; SCAN_MAR25=""; SCAN_MAR50=""
SUN_MAP25=""; SUN_MAP50=""; SUN_MAR25=""; SUN_MAR50=""
KITTI_3D_E=""; KITTI_3D_M=""; KITTI_3D_H=""; KITTI_BEV_E=""; KITTI_BEV_M=""; KITTI_BEV_H=""

if [[ -n "$SCAN_CFG" && -n "$SCAN_CKPT" ]] && ensure_file "$SCAN_CFG" && ensure_file "$SCAN_CKPT"; then
  LOG_SCAN="${LOG_DIR}/eval_scannet.log"
  echo "[grade.sh] Evaluating ScanNet:"
  if [[ "$GPUS" -gt 1 ]]; then
    run_eval_distributed "$SCAN_CFG" "$SCAN_CKPT" "$GPUS" "$LOG_SCAN"
  else
    run_eval_single_gpu "$SCAN_CFG" "$SCAN_CKPT" "$LOG_SCAN"
  fi
  read SCAN_MAP25 SCAN_MAP50 SCAN_MAR25 SCAN_MAR50 < <(parse_indoor_metrics "$LOG_SCAN")
fi

if [[ -n "$SUN_CFG" && -n "$SUN_CKPT" ]] && ensure_file "$SUN_CFG" && ensure_file "$SUN_CKPT"; then
  LOG_SUN="${LOG_DIR}/eval_sunrgbd.log"
  echo "[grade.sh] Evaluating SUN RGB-D:"
  if [[ "$GPUS" -gt 1 ]]; then
    run_eval_distributed "$SUN_CFG" "$SUN_CKPT" "$GPUS" "$LOG_SUN"
  else
    run_eval_single_gpu "$SUN_CFG" "$SUN_CKPT" "$LOG_SUN"
  fi
  read SUN_MAP25 SUN_MAP50 SUN_MAR25 SUN_MAR50 < <(parse_indoor_metrics "$LOG_SUN")
fi

if [[ -n "$KITTI_CFG" && -n "$KITTI_CKPT" ]] && ensure_file "$KITTI_CFG" && ensure_file "$KITTI_CKPT"; then
  LOG_KITTI="${LOG_DIR}/eval_kitti.log"
  echo "[grade.sh] Evaluating KITTI:"
  if [[ "$GPUS" -gt 1 ]]; then
    run_eval_distributed "$KITTI_CFG" "$KITTI_CKPT" "$GPUS" "$LOG_KITTI"
  else
    run_eval_single_gpu "$KITTI_CFG" "$KITTI_CKPT" "$LOG_KITTI"
  fi
  read KITTI_3D_E KITTI_3D_M KITTI_3D_H KITTI_BEV_E KITTI_BEV_M KITTI_BEV_H < <(parse_kitti_metrics "$LOG_KITTI")
fi

#
# 3) Update task_description.md
#
echo "[grade.sh] Updating ${OUTPUT_MD} with evaluated metrics..."
SCAN_MAP25="$SCAN_MAP25" \
SCAN_MAP50="$SCAN_MAP50" \
SUN_MAP25="$SUN_MAP25" \
SUN_MAP50="$SUN_MAP50" \
KITTI_3D_E="$KITTI_3D_E" \
KITTI_3D_M="$KITTI_3D_M" \
KITTI_3D_H="$KITTI_3D_H" \
KITTI_BEV_E="$KITTI_BEV_E" \
KITTI_BEV_M="$KITTI_BEV_M" \
KITTI_BEV_H="$KITTI_BEV_H" \
update_markdown "$OUTPUT_MD"

echo "[grade.sh] Done."