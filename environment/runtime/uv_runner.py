from __future__ import annotations

import shlex
import platform
from pathlib import Path
from typing import Dict, List, Optional


def get_shell_command() -> List[str]:
    """Get the appropriate shell command for the current platform."""
    if platform.system() == "Windows":
        # Use cmd on Windows
        return ["cmd", "/c"]
    else:
        # Use bash on Unix-like systems
        return ["bash", "-lc"]

def get_platform_commands() -> dict:
    """Get platform-specific command implementations."""
    if platform.system() == "Windows":
        return {
            "mkdir": "if not exist {path} mkdir {path}",
            "check_dir": "if exist {path}",
            "check_file": "if exist {file}",
            "remove_dir": "if exist {path} rmdir /s /q {path}",
            "export_var": "set {var}={value}",
            "export_path": "set PATH={value};%PATH%",
            "python_path": "Scripts\\python.exe",
            "bin_path": "Scripts"
        }
    else:
        return {
            "mkdir": "mkdir -p {path}",
            "check_dir": "[ -d {path} ]",
            "check_file": "[ -f {file} ]",
            "remove_dir": "rm -rf {path}",
            "export_var": "export {var}={value}",
            "export_path": 'export PATH="{value}:$PATH"',
            "python_path": "bin/python",
            "bin_path": "bin"
        }


def detect_task_overlay(task_dir: Path) -> Dict[str, str | None]:
    """Detect task-provided environment overlays.

    Supported:
    - requirements.txt (task root)
    - setup.py (editable install)
    
    """
    root_requirements = task_dir / "requirements.txt"
    setup_py = task_dir / "setup.py"
    install_sh = task_dir / "install.sh"

    return {
        "requirements": str(root_requirements) if root_requirements.exists() else None,
        "setup_py": str(setup_py) if setup_py.exists() else None,
        "install_sh": str(install_sh) if install_sh.exists() else None,
    }


def _quote(path: Path | str) -> str:
    """Quote a path appropriately for the current platform."""
    path_str = str(path)
    if platform.system() == "Windows":
        # For Windows CMD, we need to handle spaces and special characters
        # Simple approach: wrap in double quotes if it contains spaces
        if " " in path_str:
            return f'"{path_str}"'
        return path_str
    else:
        # Use shlex.quote for Unix-like systems
        return shlex.quote(path_str)


def plan_uv_commands(
    cache_dir: Path,
    venv_name: str,
    task_overlay: Dict[str, str | None],
    project_root: Path,
    command: List[str],
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict:
    venv_path = cache_dir / venv_name
    cmds: List[str] = []
    platform_cmds = get_platform_commands()

    # ensure cache dir and create venv
    cmds.append(platform_cmds["mkdir"].format(path=_quote(cache_dir)))
    
    # If a stale directory exists that isn't a real venv, remove it first
    if platform.system() == "Windows":
        # Windows CMD syntax (use parentheses to avoid ambiguous nested IFs)
        cmds.append(
            f"if exist {_quote(venv_path)} ( if not exist {_quote(venv_path / 'pyvenv.cfg')} {platform_cmds['remove_dir'].format(path=_quote(venv_path))} )"
        )
    else:
        # Unix syntax
        cmds.append(
            f"if [ -d {_quote(venv_path)} ] && [ ! -f {_quote(venv_path / 'pyvenv.cfg')} ]; then rm -rf {_quote(venv_path)}; fi"
        )
    
    # Always create a fresh virtualenv to avoid stale packages from prior runs
    # (e.g., CPU-only torch persisting when GPU becomes available).
    cmds.append(f"uv venv --clear {_quote(venv_path)}")
    # Export venv for child processes so 'python3' and 'pip' resolve inside the venv
    cmds.append(platform_cmds["export_var"].format(var="VIRTUAL_ENV", value=_quote(venv_path)))
    cmds.append(platform_cmds["export_path"].format(value=f"{_quote(venv_path)}/{platform_cmds['bin_path']}"))

    # Inject additional environment variables requested by the caller
    if env_vars:
        for key, value in env_vars.items():
            value_str = str(value)
            if platform.system() == "Windows":
                safe_value = value_str.replace('"', '""')
                cmds.append(f'set "{key}={safe_value}"')
            else:
                cmds.append(f"export {key}={shlex.quote(value_str)}")

    # Ensure apply_patch/applypatch wrappers exist on Windows so Git Bash can find them
    if platform.system() == "Windows":
        try:
            ap_py = _quote(project_root / "apply_patch.py")
            cmds.append('echo @echo off > "%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd"')
            cmds.append(f'echo "%VIRTUAL_ENV%\\Scripts\\python.exe" {ap_py} %* >> "%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd"')
            cmds.append('copy /Y "%VIRTUAL_ENV%\\Scripts\\apply_patch.cmd" "%VIRTUAL_ENV%\\Scripts\\applypatch.cmd" >nul')
            # Also create bash-friendly wrappers without extension for Git Bash resolution
            cmds.append('echo #!/usr/bin/env bash > "%VIRTUAL_ENV%\\Scripts\\apply_patch"')
            cmds.append(f'echo "%VIRTUAL_ENV%\\Scripts\\python.exe" {ap_py} "$@" >> "%VIRTUAL_ENV%\\Scripts\\apply_patch"')
            cmds.append('copy /Y "%VIRTUAL_ENV%\\Scripts\\apply_patch" "%VIRTUAL_ENV%\\Scripts\\applypatch" >nul')
        except Exception:
            pass

    venv_python = venv_path / platform_cmds["python_path"]

    # Install agent/project requirements first
    if project_root:
        agent_requirements = project_root / "requirements.txt"
        if agent_requirements.exists():
            install_py = Path(__file__).parent.parent / "scripts" / "install_requirements.py"
            install_sh = Path(__file__).parent.parent / "scripts" / "install_requirements.sh"
            if install_py.exists():
                cmds.append(
                    f"{_quote(venv_python)} {_quote(install_py)} {_quote(agent_requirements)} {_quote(venv_python)}"
                )
            elif install_sh.exists() and platform.system() != "Windows":
                cmds.append(
                    f"{_quote(install_sh)} {_quote(agent_requirements)} {_quote(venv_python)}"
                )
            else:
                # Fallback to filtering + uv pip
                filtered_reqs = cache_dir / f"{venv_name}_requirements_filtered.txt"
                with open(agent_requirements, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                filtered_lines: List[str] = []
                for raw in lines:
                    line = raw.strip()
                    if line and not line.startswith('#'):
                        lowered = line.lower()
                        if any(gpu_pkg in lowered for gpu_pkg in [
                            'nvidia-', 'cuda', 'cublas', 'cudnn', 'nccl', 'onnxruntime-gpu', 'torch+cu', 'tensorflow-gpu', 'triton'
                        ]):
                            continue
                    filtered_lines.append(line)

                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with open(filtered_reqs, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(filtered_lines))
                except (OSError, PermissionError):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='_requirements_filtered.txt', delete=False, encoding='utf-8') as f:
                        f.write('\n'.join(filtered_lines))
                        filtered_reqs = Path(f.name)

                cmds.append(
                    f"uv pip install -r {_quote(filtered_reqs)} --python {_quote(venv_python)}"
                )

    # Install task-specific requirements if present (root requirements.txt)
    if task_overlay.get("requirements"):
        install_py = Path(__file__).parent.parent / "scripts" / "install_requirements.py"
        install_sh = Path(__file__).parent.parent / "scripts" / "install_requirements.sh"
        if install_py.exists():
            cmds.append(
                f"{_quote(venv_python)} {_quote(install_py)} {_quote(task_overlay['requirements'])} {_quote(venv_python)}"
            )
        elif install_sh.exists() and platform.system() != "Windows":
            cmds.append(
                f"{_quote(install_sh)} {_quote(task_overlay['requirements'])} {_quote(venv_python)}"
            )
        else:
            # Fallback to regular pip install
            cmds.append(
                f"uv pip install -r {_quote(task_overlay['requirements'])} --python {_quote(venv_python)}"
            )

    # Editable install if setup.py provided
    if task_overlay.get("setup_py"):
        task_dir = Path(task_overlay["setup_py"]).parent
        cmds.append(
            f"cd {_quote(task_dir)} && {_quote(venv_python)} -m pip install -e ."
        )

    # Note: environment.yml is not automatically processed under uv.

    # Use venv python for the final run command, if provided
    run_cmd: List[str] = []
    if command:
        try:
            exe_name = Path(command[0]).name.lower()
        except Exception:
            exe_name = ""
        if exe_name.startswith("python"):
            run_cmd = [str(venv_python)] + command[1:]
        else:
            run_cmd = command

    quoted_run_cmd = [_quote(arg) for arg in run_cmd] if run_cmd else []
    shell_parts: List[str] = cmds.copy()
    # Append the actual program execution last
    if quoted_run_cmd:
        shell_parts.append(" ".join(quoted_run_cmd))

    if platform.system() == "Windows":
        # Emit a .cmd script with step markers and fail-fast behavior.
        # This ensures each run starts from a fresh, self-contained script so
        # stale environment from prior runs cannot leak.
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        script_path = cache_dir / f"{venv_name}_run.cmd"

        try:
            lines: List[str] = []
            # Hide command echo to avoid leaking absolute paths in logs; step markers still printed explicitly.
            lines.append("@echo off")
            lines.append("setlocal")
            # Ensure Python uses UTF-8 for reliable logging
            lines.append("set PYTHONIOENCODING=utf-8")
            lines.append("set PYTHONUTF8=1")
            for i, part in enumerate(shell_parts, start=1):
                # Provide a stable marker so upstream can inject env between setlocal and first step
                lines.append(f"echo [RG step {i}]")
                lines.append(part)
                lines.append("if errorlevel 1 exit /b 1")
            lines.append("exit /b 0")
            script_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            # Fall back to direct shell join if writing fails
            return {
                "venv_path": str(venv_path),
                "overlay": task_overlay,
                "exec": run_cmd,
                "shell": get_shell_command() + [" && ".join(shell_parts)],
            }

        return {
            "venv_path": str(venv_path),
            "overlay": task_overlay,
            "exec": run_cmd,
            "shell": ["cmd", "/c", str(script_path)],
        }
    else:
        return {
            "venv_path": str(venv_path),
            "overlay": task_overlay,
            "exec": run_cmd,
            "shell": get_shell_command() + [" && ".join(shell_parts)],
        }


