#!/usr/bin/env python3
"""
GPU Detection Utility for ResearchGym

This script detects available GPU hardware and returns appropriate
package installation commands for CPU-only or GPU-enabled environments.
"""

import os
import shutil
import subprocess
import sys
import json
from typing import Dict, List, Optional, Tuple


def _find_nvidia_smi() -> Optional[str]:
    """Locate nvidia-smi across common install paths and PATH.

    Returns the absolute path if found, else None.
    """
    candidates: List[str] = []
    # Prefer PATH
    path_hit = shutil.which('nvidia-smi') if 'shutil' in globals() else None
    if path_hit:
        candidates.append(path_hit)
    # Windows typical locations
    program_files = os.environ.get('ProgramFiles', r'C:\\Program Files')
    candidates += [
        os.path.join(program_files, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'),
        os.path.join(os.environ.get('SystemRoot', r'C:\\Windows'), 'System32', 'nvidia-smi.exe'),
    ]
    # Linux typical locations
    candidates += [
        '/usr/bin/nvidia-smi',
        '/usr/local/nvidia/bin/nvidia-smi',
        '/bin/nvidia-smi',
    ]
    for c in candidates:
        try:
            if c and os.path.exists(c):
                return c
        except Exception:
            continue
    return None


def _try_powershell_gpu_probe() -> bool:
    """Fallback GPU probe on Windows via PowerShell CIM if nvidia-smi isn't found."""
    try:
        ps = shutil.which('powershell') or 'powershell'
        result = subprocess.run(
            [ps, '-NoProfile', '-Command', 'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
            capture_output=True, text=True, timeout=8
        )
        if result.returncode == 0 and 'nvidia' in result.stdout.lower():
            return True
    except Exception:
        pass
    return False


def check_nvidia_gpu() -> Tuple[bool, Optional[str]]:
    """Check if NVIDIA GPU is available and get CUDA version.

    Tries nvidia-smi from PATH and common install paths, then falls back to a
    Windows CIM probe. CUDA version is best-effort; defaults to a safe wheel
    series when unknown.
    """
    try:
        smi = _find_nvidia_smi() or 'nvidia-smi'
        result = subprocess.run([smi, '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=8)
        if result.returncode == 0:
            # Try to get CUDA version via nvcc or assume a recent compatible version
            try:
                nvcc = shutil.which('nvcc') or 'nvcc'
                cuda_result = subprocess.run([nvcc, '--version'], capture_output=True, text=True, timeout=5)
                if cuda_result.returncode == 0:
                    for line in cuda_result.stdout.split('\n'):
                        if 'Cuda compilation tools' in line and 'V' in line:
                            version = line.split('V')[-1].strip()
                            return True, version
            except Exception:
                pass
            # Use a broadly supported default if nvcc not available
            return True, '12.6'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # Windows fallback using CIM if nvidia-smi isn't available
    # try:
    #     import platform
    #     if platform.system() == 'Windows' and _try_powershell_gpu_probe():
    #         return True, '12.1'
    # except Exception:
    #     pass
    
    # Check if we're inside a Docker container with GPU support
    try:
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                # Check for NVIDIA container runtime
                result = subprocess.run(['ls', '/dev/nvidia0'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    return True, "11.8"  # Assume CUDA available in GPU container
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return False, None


def check_metal_gpu() -> bool:
    """Check if Apple Metal GPU is available (for M1/M2 Macs)."""
    try:
        import platform
        if platform.system() == 'Darwin':
            # Check for Apple Silicon
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            if result.returncode == 0 and 'arm64' in result.stdout:
                return True
    except Exception:
        pass
    return False


def get_torch_install_command(gpu_type: str = "none", cuda_version: Optional[str] = None) -> List[str]:
    """Get appropriate PyTorch installation command.

    Returns arguments in the correct order for pip/uv pip: options first,
    followed by packages.
    """
    if gpu_type == "nvidia":
        # Default to a modern CUDA wheel series when version is unknown
        if cuda_version:
            major_minor = ".".join(cuda_version.split(".")[:2])
            cuda_suffix = major_minor.replace(".", "")
        else:
            cuda_suffix = "121"
        return [
            "--index-url", f"https://download.pytorch.org/whl/cu{cuda_suffix}",
            "torch>=2.0.0,<3.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ]
    elif gpu_type == "metal":
        return [
            "torch>=2.0.0,<3.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ]
    else:
        return [
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch>=2.0.0,<3.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ]


def get_additional_packages(gpu_type: str = "none") -> Dict[str, List[str]]:
    """Get additional GPU-specific or CPU-specific packages."""
    packages = {
        "install": [],
        "skip": []
    }
    
    if gpu_type == "nvidia":
        packages["install"].extend([
            "nvidia-ml-py3",  # For GPU monitoring
        ])
        # Don't skip NVIDIA packages - let PyTorch installation handle them
    elif gpu_type == "metal":
        # Skip NVIDIA-specific packages for Apple Silicon
        packages["skip"].extend([
            "nvidia-*",
            "*cuda*", 
            "*cublas*",
            "*cudnn*",
            "*cufft*",
            "*curand*",
            "*cusolver*",
            "*cusparse*",
            "*nccl*",
            "*nvjitlink*",
            "*nvtx*",
            "triton",
            "onnxruntime-gpu"
        ])
        packages["install"].extend([
            "onnxruntime",  # CPU version instead of GPU
        ])
    else:
        # Skip GPU-specific packages for CPU-only installations
        packages["skip"].extend([
            "nvidia-*",
            "*cuda*", 
            "*cublas*",
            "*cudnn*",
            "*cufft*",
            "*curand*",
            "*cusolver*",
            "*cusparse*",
            "*nccl*",
            "*nvjitlink*",
            "*nvtx*",
            "triton",
            "onnxruntime-gpu"
        ])
        packages["install"].extend([
            "onnxruntime",  # CPU version instead of GPU
        ])
    
    return packages


def main():
    """Main detection and recommendation logic."""
    # Detect available GPU
    has_nvidia, cuda_version = check_nvidia_gpu()
    has_metal = check_metal_gpu()
    
    gpu_info = {
        "has_nvidia": has_nvidia,
        "cuda_version": cuda_version,
        "has_metal": has_metal,
        "detected_type": "none"
    }
    
    if has_nvidia:
        gpu_info["detected_type"] = "nvidia"
    elif has_metal:
        gpu_info["detected_type"] = "metal"
    
    # Get installation recommendations
    torch_cmd = get_torch_install_command(gpu_info["detected_type"], cuda_version)
    additional_packages = get_additional_packages(gpu_info["detected_type"])
    
    result = {
        "gpu_info": gpu_info,
        "torch_install": torch_cmd,
        "additional_packages": additional_packages,
        "environment_vars": {}
    }
    
    # Set environment variables based on detection
    if gpu_info["detected_type"] == "nvidia":
        result["environment_vars"]["CUDA_VISIBLE_DEVICES"] = "0"
        result["environment_vars"]["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6;9.0"
    elif gpu_info["detected_type"] == "metal":
        result["environment_vars"]["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        result["environment_vars"]["CUDA_VISIBLE_DEVICES"] = ""
    
    # Output format depends on how script is called
    if len(sys.argv) > 1:
        if sys.argv[1] == "--json":
            print(json.dumps(result, indent=2))
        elif sys.argv[1] == "--torch-cmd":
            print(" ".join(torch_cmd))
        elif sys.argv[1] == "--gpu-type":
            print(gpu_info["detected_type"])
        elif sys.argv[1] == "--has-gpu":
            print("true" if gpu_info["detected_type"] != "none" else "false")
        else:
            print(f"GPU Type: {gpu_info['detected_type']}")
            if cuda_version:
                print(f"CUDA Version: {cuda_version}")
            print(f"PyTorch Install: {' '.join(torch_cmd)}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
