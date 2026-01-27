# Grading script for AIGI Detection task
# This script evaluates a submission produced by an agent and fills out the
# metrics required by tasks/test/aigi-detection/task_description.md.

# ============================
# Usage
#   bash grade.sh /absolute/path/to/submission_dir [--device cpu|cuda] [--max-sample N]
#
# Expected submission directory layout:
#   submission/
#     deepfake/
#       weights.pth                
#       detector.yaml             
#     ufd/
#       ckpt.pth                   
#
# Dataset locations:
#   DeepfakeBench:
#     DFB_DATA_ROOT   -> defaults to ./DeepfakeBench/training/datasets
#     DFB_JSON_ROOT   -> defaults to ./DeepfakeBench/preprocessing/dataset_json
#   UniversalFakeDetect:
#     UFD_DATA_ROOT
# ============================

# set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$ROOT_DIR"
DFB_DIR="$ROOT_DIR/DeepfakeBench/training"
UFD_DIR="$ROOT_DIR/UniversalFakeDetect_Benchmark"

# SUBMISSION_DIR="${1:-}"
SUBMISSION_DIR=""
DEVICE="cuda"
MAX_SAMPLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"; shift 2;;
    --max-sample)
      MAX_SAMPLE="$2"; shift 2;;
    *)
      if [[ -z "${SUBMISSION_DIR}" ]]; then
        SUBMISSION_DIR="$1"; shift 1
      else
        echo "Unknown arg: $1" >&2; exit 2
      fi
      ;;
  esac
done

if [[ -z "${SUBMISSION_DIR}" ]]; then
  echo "Usage: bash grade.sh /absolute/path/to/submission_dir [--device cpu|cuda] [--max-sample N]" >&2
  exit 2
fi

if [[ ! -d "${SUBMISSION_DIR}" ]]; then
  echo "Submission dir not found: ${SUBMISSION_DIR}" >&2
  exit 2
fi

# Results
OUT_DIR="$SUBMISSION_DIR/.grade"
mkdir -p "$OUT_DIR"

# Pin device for child processes
export CUDA_VISIBLE_DEVICES="0"
if [[ "$DEVICE" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

# Helper: run python in task venv if present
PYTHON_BIN="python3"
if [[ -x "$TASK_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$TASK_DIR/.venv/bin/python"
fi

# Helper: JSON writer
write_json() {
  local path="$1"; shift
  printf '%s' "$*" > "$path"
}


# DeepfakeBench evaluation

run_deepfakebench() {
  local weights_path detector_yaml
  local df_sub="$SUBMISSION_DIR/deepfake"
  local cfg_default="$DFB_DIR/config/detector/method.yaml"
  local cfg_override="$df_sub/detector.yaml"

  if [[ -f "$df_sub/weights.pth" ]]; then
    weights_path="$df_sub/weights.pth"
  elif [[ -f "$df_sub/model.pth" ]]; then
    weights_path="$df_sub/model.pth"
  else
    # Not fatal: skip DeepfakeBench if no weights provided
    echo "[DeepfakeBench] No weights found in $df_sub; skipping." >&2
    return 0
  fi

  if [[ -f "$cfg_override" ]]; then
    detector_yaml="$cfg_override"
  else
    detector_yaml="$cfg_default"
  fi

  # Ensure env roots exist or fall back to defaults inside repo
  export DFB_DATA_ROOT="${DFB_DATA_ROOT:-$ROOT_DIR/DeepfakeBench/training/datasets}"
  export DFB_JSON_ROOT="${DFB_JSON_ROOT:-$ROOT_DIR/DeepfakeBench/preprocessing/dataset_json}"

  pushd "$DFB_DIR" >/dev/null

  # Protocol-1: Cross-dataset AUC (video-level)
  local p1_datasets=("Celeb-DF-v2" "DeepFakeDetection" "DFDC" "DFDCP" "DeeperForensics-1.0" "WDF" "FFIW")
  # # Test one dataset at a time - modify this line for individual testing
  # local p1_datasets=("Celeb-DF-v2")
  local p1_log="$OUT_DIR/dfb_protocol1.log"
  : > "$p1_log"
  local p1_json="$OUT_DIR/dfb_protocol1.json"

  local tmp_json_p1="$OUT_DIR/dfb_p1_tmp.json"
  echo '{"results":{}}' > "$tmp_json_p1"

  for ds in "${p1_datasets[@]}"; do
    # test.py prints per-dataset metrics; we capture and parse video_auc
    local tmp_out="$OUT_DIR/dfb_${ds}_out.txt"
    $PYTHON_BIN test.py \
      --detector_path "$detector_yaml" \
      --test_dataset "$ds" \
      --weights_path "$weights_path" | tee "$tmp_out"

    # Parse video-level AUC from output (metrics.utils prints keys including 'video_auc')
    local vauc
    vauc=$(grep -E "video_auc" -m1 "$tmp_out" | sed -E 's/.*video_auc: *([0-9.]+)/\1/g' || true)
    if [[ -z "$vauc" ]]; then
      # fallback: use auc if video_auc missing
      vauc=$(grep -E "[^a-z]auc: *[0-9.]+" -m1 "$tmp_out" | sed -E 's/.*auc: *([0-9.]+)/\1/g' || echo "NaN")
    fi

    echo "$ds: $vauc" | tee -a "$p1_log"
  done

  {
    echo '{'
    echo '  "protocol": "cross-dataset",'
    echo '  "metric": "video_auc",'
    echo '  "datasets": {'
    local first=1
    for ds in "${p1_datasets[@]}"; do
      local vauc_line=$(grep -E "^$ds:" "$p1_log" | tail -n1 | awk '{print $2}')
      if [[ "$first" -eq 1 ]]; then first=0; else echo ','; fi
      printf '    "%s": %s' "$ds" "${vauc_line:-null}"
    done
    echo ''
    echo '  },'
    # compute avg via awk
    local avg
    avg=$(awk '{sum+=$2; n+=1} END {if(n>0) printf("%.6f", sum/n); else print "null"}' "$p1_log" 2>/dev/null || echo null)
    printf '  "avg": %s\n' "${avg:-null}"
    echo '}'
  } > "$p1_json"

  # Protocol-2: DF40 cross-method AUC
  local p2_datasets=("uniface_ff" "blendface_ff" "mobileswap_ff" "e4s_Fake" "danet_Fake" "fsgan_Fake" "inswap_Fake" "simswap_Fake")
  # local p2_datasets=()
  local p2_log="$OUT_DIR/dfb_protocol2.log"
  : > "$p2_log"
  local p2_json="$OUT_DIR/dfb_protocol2.json"

  for ds in "${p2_datasets[@]}"; do
    local tmp_out="$OUT_DIR/dfb_${ds}_out.txt"
    $PYTHON_BIN test.py \
      --detector_path "$detector_yaml" \
      --test_dataset "$ds" \
      --weights_path "$weights_path" | tee "$tmp_out"
    local vauc
    vauc=$(grep -E "video_auc" -m1 "$tmp_out" | sed -E 's/.*video_auc: *([0-9.]+)/\1/g' || true)
    if [[ -z "$vauc" ]]; then
      vauc=$(grep -E "[^a-z]auc: *[0-9.]+" -m1 "$tmp_out" | sed -E 's/.*auc: *([0-9.]+)/\1/g' || echo "NaN")
    fi
    echo "$ds: $vauc" | tee -a "$p2_log"
  done

  {
    echo '{'
    echo '  "protocol": "df40-cross-method",'
    echo '  "metric": "video_auc",'
    echo '  "datasets": {'
    local first=1
    for ds in "${p2_datasets[@]}"; do
      local vauc_line=$(grep -E "^$ds:" "$p2_log" | tail -n1 | awk '{print $2}')
      if [[ "$first" -eq 1 ]]; then first=0; else echo ','; fi
      printf '    "%s": %s' "$ds" "${vauc_line:-null}"
    done
    echo ''
    echo '  },'
    local avg
    avg=$(awk '{sum+=$2; n+=1} END {if(n>0) printf("%.6f", sum/n); else print "null"}' "$p2_log" 2>/dev/null || echo null)
    printf '  "avg": %s\n' "${avg:-null}"
    echo '}'
  } > "$p2_json"

  popd >/dev/null
}

# UniversalFakeDetect evaluation

run_ufd() {
  local ufd_sub="$SUBMISSION_DIR/ufd"
  local ckpt
  if [[ -f "$ufd_sub/ckpt.pth" ]]; then
    ckpt="$ufd_sub/ckpt.pth"
  elif [[ -f "$ufd_sub/model.pth" ]]; then
    ckpt="$ufd_sub/model.pth"
  else
    echo "[UFD] No ckpt found in $ufd_sub; skipping." >&2
    return 0
  fi

  if [[ -z "${UFD_DATA_ROOT:-}" || ! -d "${UFD_DATA_ROOT}" ]]; then
    echo "[UFD] UFD_DATA_ROOT not set or missing; skipping UFD evaluation." >&2
    return 0
  fi

  pushd "$UFD_DIR" >/dev/null
  local res_dir="$OUT_DIR/ufd_results"
  mkdir -p "$res_dir"

  local extra=()
  if [[ -n "$MAX_SAMPLE" ]]; then
    extra+=("--max_sample" "$MAX_SAMPLE")
  fi

  PYTHONPATH="$UFD_DIR":$PYTHONPATH \
  $PYTHON_BIN validate.py \
    --arch CLIP:ViT-L/14 \
    --ckpt "$ckpt" \
    --result_folder "$res_dir" \
    --batch_size 128 "${extra[@]}"

  # Parse ap.txt and acc0.txt to compute mAP and mAcc
  local map_json="$OUT_DIR/ufd_map.json"
  local macc_json="$OUT_DIR/ufd_macc.json"

  # ap.txt lines: key: value
  if [[ -f "$res_dir/ap.txt" ]]; then
    local map
    map=$(awk -F': ' '/^[a-zA-Z0-9_-]+: /{sum+=$2; n+=1} END {if(n>0) printf("%.2f", sum/n);}' "$res_dir/ap.txt" || true)
    echo "{\"metric\":\"mAP\",\"value\":${map:-null}}" > "$map_json"
  fi

  # acc0.txt lines: key: r_acc f_acc acc
  if [[ -f "$res_dir/acc0.txt" ]]; then
    local macc
    macc=$(awk -F': ' '/^[a-zA-Z0-9_-]+: /{split($2,a,"  "); sum+=a[3]; n+=1} END {if(n>0) printf("%.2f", sum/n);}' "$res_dir/acc0.txt" || true)
    echo "{\"metric\":\"mAcc\",\"value\":${macc:-null}}" > "$macc_json"
  fi

  popd >/dev/null
}


# Adapter-based PEFT (frame-level AUC)


run_peft_frame_auc() {
  # If the agent provides per-dataset frame-level numbers, we can read them from a
  # conventional file submission/peft/frame_auc.json; otherwise skip.
  local peft_json_in="$SUBMISSION_DIR/peft/frame_auc.json"
  local out="$OUT_DIR/peft_frame_auc.json"
  if [[ -f "$peft_json_in" ]]; then
    cp "$peft_json_in" "$out"
  fi
}

# ============================
# Aggregate to final outputs
# ============================

aggregate_outputs() {
  local final_json="$SUBMISSION_DIR/results.json"
  local final_md="$SUBMISSION_DIR/results.md"

  # Collect intermediate files if exist
  local p1="$OUT_DIR/dfb_protocol1.json"
  local p2="$OUT_DIR/dfb_protocol2.json"
  local map="$OUT_DIR/ufd_map.json"
  local macc="$OUT_DIR/ufd_macc.json"
  local peft="$OUT_DIR/peft_frame_auc.json"

  # Build JSON
  {
    echo '{'
    echo '  "deepfake_protocol_1":',
    if [[ -f "$p1" ]]; then cat "$p1"; else echo 'null'; fi
    echo ','
    echo '  "deepfake_protocol_2":',
    if [[ -f "$p2" ]]; then cat "$p2"; else echo 'null'; fi
    echo ','
    echo '  "ufd_map":',
    if [[ -f "$map" ]]; then cat "$map"; else echo 'null'; fi
    echo ','
    echo '  "ufd_macc":',
    if [[ -f "$macc" ]]; then cat "$macc"; else echo 'null'; fi
    echo ','
    echo '  "peft_frame_auc":',
    if [[ -f "$peft" ]]; then cat "$peft"; else echo 'null'; fi
    echo ''
    echo '}'
  } > "$final_json"

  # Build Markdown skeleton matching task_description tables
  {
    echo '## Deepfake Detection — Protocol-1: Cross-dataset (video-level AUC)'
    if [[ -f "$p1" ]]; then
      # Simple printer using jq if available
      if command -v jq >/dev/null 2>&1; then
        jq -r '
          .datasets as $d |
          ("| Dataset | AUC |"),
          ("|---|---|"),
          ($d | to_entries[] | "| \(.key) | \(.value) |") ,
          ("\nAvg: " + (.avg|tostring))
        ' "$p1"
      fi
    else
      echo '_No results_'
    fi
    echo ''

    echo '## Deepfake Detection — Protocol-2: Cross-method on DF40 (video-level AUC)'
    if [[ -f "$p2" ]]; then
      if command -v jq >/dev/null 2>&1; then
        jq -r '
          .datasets as $d |
          ("| Method | AUC |"),
          ("|---|---|"),
          ($d | to_entries[] | "| \(.key) | \(.value) |"),
          ("\nAvg: " + (.avg|tostring))
        ' "$p2"
      fi
    else
      echo '_No results_'
    fi
    echo ''

    echo '## Synthetic Image Detection — UniversalFakeDetect'
    if [[ -f "$map" ]]; then
      local v=$(jq -r '.value' "$map" 2>/dev/null || echo "null")
      echo "mAP: ${v}"
    else
      echo 'mAP: N/A'
    fi
    if [[ -f "$macc" ]]; then
      local v=$(jq -r '.value' "$macc" 2>/dev/null || echo "null")
      echo "mAcc: ${v}"
    else
      echo 'mAcc: N/A'
    fi
    echo ''

    echo '## Adapter-based PEFT baselines on deepfake (frame-level AUC)'
    if [[ -f "$peft" ]]; then
      cat "$peft"
    else
      echo '_No results provided by submission_'
    fi
  } > "$final_md"

  echo "Wrote $final_json and $final_md"
}

# Main

echo "[Grader] Starting evaluation for submission: $SUBMISSION_DIR"

run_deepfakebench || true
run_ufd || true
run_peft_frame_auc || true
aggregate_outputs

echo "[Grader] Done."

# Minimal markdown updater: update task_description.md "Your Method" rows.
# First fill replaces placeholder; subsequent runs append a new row below existing populated one.
MD_PATH="$TASK_DIR/task_description.md"
if [[ -f "$MD_PATH" ]]; then
  python - <<'PY'
import json, re, sys
from pathlib import Path

task_md = Path(sys.argv[1])
results_json = Path(sys.argv[2])

def load_avg(path, key):
    try:
        obj = json.loads(path.read_text())
        x = obj.get(key)
        if isinstance(x, dict):
            return x.get('avg', None)
        return None
    except Exception:
        return None

md = task_md.read_text(encoding='utf-8')

# Load aggregates if present
try:
    obj = json.loads(Path(results_json).read_text())
except Exception:
    obj = {}

def fmt(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "--"

# Sections mapping: keep very minimal; if a value is missing, keep '--'
def update_section(md, header, num_cols):
    # Find header line
    idx = md.find(header)
    if idx == -1:
        return md
    # Find the block until next blank line or next '###'
    end = md.find('\n\n', idx)
    end = end if end != -1 else len(md)
    block = md[idx:end]

    # Build row with placeholders (we don't compute per-dataset cells here)
    values = ['--'] * num_cols
    row = '| Your Method | ' + ' | '.join(values) + ' |\n'

    # locate existing 'Your Method' row
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith('| your method |'):
            # replace placeholder if no digits; else append a new row
            if re.search(r'\d', ln):
                lines.insert(i + 1, row.rstrip('\n'))
            else:
                lines[i] = row.rstrip('\n')
            new_block = '\n'.join(lines)
            return md[:idx] + new_block + md[end:]
    # If not found, append row at end of block
    if not block.endswith('\n'):
        block += '\n'
    block += row
    return md[:idx] + block + md[end:]

# Protocol-1 table: 8 datasets + Avg = 9 numeric columns
md = update_section(md, '### Deepfake Detection — Protocol-1', 9)
md = update_section(md, '### Deepfake Detection — Protocol-2', 9)
# UFD tables each have 17 dataset cols + one summary metric col
md = update_section(md, '### Synthetic Image Detection — UniversalFakeDetect (mAP)', 17)
md = update_section(md, '### Synthetic Image Detection — UniversalFakeDetect (mAcc)', 17)
# PEFT table: 3 datasets + Avg = 4 cols
md = update_section(md, '### Adapter-based PEFT baselines on deepfake', 4)

task_md.write_text(md, encoding='utf-8')
print('[Grader] Updated task_description.md Your Method rows (append after first write).')
PY
  "$MD_PATH" "$SUBMISSION_DIR/results.json"
fi

