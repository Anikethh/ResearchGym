#!/bin/bash
set -e

REQUIREMENTS_FILE="$1"
PYTHON_CMD="${2:-python}"

if [[ -z "$REQUIREMENTS_FILE" || ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "Usage: $0 <requirements.txt> [python_cmd]"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_DETECTOR="$SCRIPT_DIR/detect_gpu.py"

# Detect GPU
GPU_TYPE=$($PYTHON_CMD "$GPU_DETECTOR" --gpu-type 2>/dev/null || echo "none")

# Create temp filtered requirements
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT
FILTERED_REQUIREMENTS="$TEMP_DIR/requirements_filtered.txt"

if [[ $GPU_TYPE == "none" ]]; then
    # Filter out GPU packages for CPU-only systems
    grep -v -E "(nvidia-|cuda|cublas|cudnn|nccl|onnxruntime-gpu|torch\+cu|tensorflow-gpu|triton)" "$REQUIREMENTS_FILE" > "$FILTERED_REQUIREMENTS" || cp "$REQUIREMENTS_FILE" "$FILTERED_REQUIREMENTS"
    
    # Replace GPU packages with CPU equivalents
    sed -i.bak 's/onnxruntime-gpu/onnxruntime/g' "$FILTERED_REQUIREMENTS" 2>/dev/null || true
elif [[ $GPU_TYPE == "metal" ]]; then
    # Filter out NVIDIA-specific packages for Apple Silicon (Metal GPU)
    # Keep general GPU packages but remove NVIDIA/CUDA specific ones
    grep -v -E "(nvidia-|cuda|cublas|cudnn|nccl|torch\+cu|tensorflow-gpu|triton)" "$REQUIREMENTS_FILE" > "$FILTERED_REQUIREMENTS" || cp "$REQUIREMENTS_FILE" "$FILTERED_REQUIREMENTS"
    
    # Replace GPU packages with CPU/Metal equivalents
    sed -i.bak 's/onnxruntime-gpu/onnxruntime/g' "$FILTERED_REQUIREMENTS" 2>/dev/null || true
else
    # Keep original requirements for GPU systems
    cp "$REQUIREMENTS_FILE" "$FILTERED_REQUIREMENTS"
fi

# Install with UV
if command -v uv >/dev/null 2>&1; then
    # Resolve absolute interpreter path for targeting the correct environment
    PYTHON_ABS=$($PYTHON_CMD -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PYTHON_CMD")
    if [[ $GPU_TYPE == "none" ]]; then
        # CPU-only: install PyTorch with CPU index first
        uv pip install --python "$PYTHON_ABS" --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio 2>/dev/null || true
    fi
    uv pip install --python "$PYTHON_ABS" -r "$FILTERED_REQUIREMENTS"
else
    # Fallback to pip if UV not available
    if [[ $GPU_TYPE == "none" ]]; then
        $PYTHON_CMD -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    fi
    $PYTHON_CMD -m pip install -r "$FILTERED_REQUIREMENTS"
fi
