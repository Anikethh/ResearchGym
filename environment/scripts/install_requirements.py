#!/usr/bin/env python3
"""
Cross-platform requirements installer for ResearchGym.

Usage:
  python install_requirements.py <requirements.txt> [python_cmd]

Notes:
- Detects GPU type (NVIDIA/Metal/none) and filters/adjusts packages.
- Uses uv pip when available; falls back to pip.
- Intended to replace the bash version for Windows compatibility.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import platform as _platform
from typing import List
import os


def run(cmd: List[str]) -> int:
    return subprocess.call(cmd)


def detect_gpu_type(scripts_dir: Path, python_cmd: str) -> str:
    detector = scripts_dir / "detect_gpu.py"
    if not detector.exists():
        return "none"
    try:
        result = subprocess.run(
            [python_cmd, str(detector), "--gpu-type"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "none"
    except Exception:
        pass
    return "none"


def filter_requirements(req_path: Path, gpu_type: str) -> str:
    lines = req_path.read_text(encoding="utf-8").splitlines()
    filtered: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            filtered.append(raw)
            continue
        lower = line.lower()
        # Windows + NVIDIA: map 'triton*' to 'triton-windows' (drop version spec)
        if _platform.system() == "Windows" and gpu_type == "nvidia" and "triton" in lower:
            filtered.append("triton-windows")
            continue
        # On Windows, unconditionally skip Linux-only packages like triton
        windows_block = ("triton" in lower) if _platform.system() == "Windows" else False
        skip_gpu = any(s in lower for s in [
            "nvidia-", "cuda", "cublas", "cudnn", "cufft", "curand",
            "cusolver", "cusparse", "nccl", "nvjitlink", "nvtx",
            "onnxruntime-gpu", "torch+cu", "tensorflow-gpu", "triton",
        ])
        if gpu_type == "nvidia" and not windows_block:
            # We proactively install CUDA-enabled torch/vision/audio; skip them here to avoid CPU re-install
            if lower.startswith("torch==") or lower.startswith("torch ") or lower == "torch":
                continue
            if lower.startswith("torchvision==") or lower.startswith("torchvision ") or lower == "torchvision":
                continue
            if lower.startswith("torchaudio==") or lower.startswith("torchaudio ") or lower == "torchaudio":
                continue
            # keep other packages as-is
            filtered.append(line)
            continue
        if windows_block or (gpu_type in ("metal", "none") and skip_gpu):
            # drop GPU-specific packages
            continue
        # On CPU/Metal, we already preinstall torch/vision/audio; avoid pin conflicts
        if gpu_type in ("metal", "none"):
            if lower.startswith("torch==") or lower.startswith("torch ") or lower == "torch":
                continue
            if lower.startswith("torchvision==") or lower.startswith("torchvision ") or lower == "torchvision":
                continue
            if lower.startswith("torchaudio==") or lower.startswith("torchaudio ") or lower == "torchaudio":
                continue
        # replace onnxruntime-gpu -> onnxruntime
        if "onnxruntime-gpu" in lower:
            filtered.append(raw.lower().replace("onnxruntime-gpu", "onnxruntime"))
        else:
            filtered.append(raw)
    from tempfile import NamedTemporaryFile

    tmp = NamedTemporaryFile("w", suffix="_requirements_filtered.txt", delete=False, encoding="utf-8")
    with tmp as f:
        f.write("\n".join(filtered))
    return tmp.name


def main() -> int:
    parser = argparse.ArgumentParser()
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("python_cmd", nargs="?", default=sys.executable)
    args = parser.parse_args()

    req = args.requirements
    if not req.exists():
        print(f"Requirements file not found: {req}", file=sys.stderr)
        return 2

    scripts_dir = Path(__file__).resolve().parent
    # Allow manual override for robustness
    override = os.environ.get("RG_FORCE_GPU", "").strip().lower()
    gpu_type = override if override in {"nvidia", "metal", "none"} else detect_gpu_type(scripts_dir, args.python_cmd)

    # CPU-only/Metal: pre-install torch from proper index (best-effort)
    # Use uv if available, otherwise pip
    uv = shutil.which("uv")
    torch_cmd: List[str] | None = None
    if gpu_type == "none":
        torch_cmd = [
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    elif gpu_type == "nvidia":
        # Proactively install CUDA-enabled wheels on NVIDIA systems (Windows included)
        try:
            from detect_gpu import check_nvidia_gpu, get_torch_install_command as _torch_cmd

            has, cuda_ver = check_nvidia_gpu()
            # Allow explicit CUDA version override via env
            cuda_override = os.environ.get("RG_CUDA_VERSION", "").strip()
            if cuda_override:
                cuda_ver = cuda_override
            torch_cmd = _torch_cmd("nvidia", cuda_ver if (has or cuda_override) else None)
        except Exception:
            # Fallback to a recent CUDA index if detection fails
            torch_cmd = [
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
                "torch",
                "torchvision",
                "torchaudio",
            ]
    elif gpu_type == "metal":
        torch_cmd = ["torch", "torchvision", "torchaudio"]

    if torch_cmd:
        if uv:
            run(["uv", "pip", "install", "--python", args.python_cmd] + torch_cmd)
        else:
            run([args.python_cmd, "-m", "pip", "install"] + torch_cmd)

    # Filter and install project requirements
    filtered = filter_requirements(req, gpu_type)
    if uv:
        rc = run(["uv", "pip", "install", "--python", args.python_cmd, "-r", filtered])
    else:
        rc = run([args.python_cmd, "-m", "pip", "install", "-r", filtered])

    # Final safeguard: ensure CUDA-enabled torch remains installed on NVIDIA.
    if gpu_type == "nvidia":
        try:
            probe = subprocess.run(
                [args.python_cmd, "-c", "import torch; import sys; sys.stdout.write(str(bool(getattr(torch, 'version', None) and getattr(torch.version, 'cuda', None) and torch.cuda.is_available())))"],
                capture_output=True, text=True, timeout=15,
            )
            has_cuda = probe.returncode == 0 and probe.stdout.strip().lower() == "true"
        except Exception:
            has_cuda = False
        if not has_cuda:
            try:
                from detect_gpu import get_torch_install_command as _torch_cmd
                # Attempt to (re)install CUDA wheels compatible with the environment
                torch_cmd = _torch_cmd("nvidia", None)
            except Exception:
                torch_cmd = [
                    "--index-url", "https://download.pytorch.org/whl/cu126",
                    "torch", "torchvision", "torchaudio",
                ]
            if uv:
                run(["uv", "pip", "install", "--python", args.python_cmd] + torch_cmd)
            else:
                run([args.python_cmd, "-m", "pip", "install"] + torch_cmd)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
