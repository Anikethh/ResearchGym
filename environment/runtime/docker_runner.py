from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import os


def plan_docker_command(
    image: str,
    run_dir: Path,
    ml_master_root: Path,
    command: List[str],
    gpus: bool = False,
    env_keys: Optional[List[str]] = None,
    prelude: Optional[str] = None,
    env_file: Optional[Path] = None,
    task_dir: Optional[Path] = None,
    user: Optional[str] = None,
    run_command_as_user: Optional[str] = None,
) -> Dict:
    # Mount run_dir as /run, ml_master_root as /agent, and workspace for clarity
    mounts = [
        {"type": "bind", "source": str(run_dir), "target": "/run"},
        {"type": "bind", "source": str(ml_master_root), "target": "/agent"},
    ]
    if task_dir is not None:
        abs_task_dir = Path(task_dir).resolve()
        mounts.append({"type": "bind", "source": str(abs_task_dir), "target": "/task"})
    # Rewrite host paths in the agent command to in-container mounts
    rewritten_cmd: List[str] = []
    for i, arg in enumerate(command):
        if isinstance(arg, str):
            # Rewrite Python executable path for Docker container
            # Matches /python, /python3, /python3.10, /python3.12, etc.
            if i == 0 and re.search(r'/python\d*(\.\d+)?$', arg):
                arg = "python"
            else:
                # Rewrite directory paths
                arg = arg.replace(str(run_dir), "/run").replace(str(ml_master_root), "/agent")
        rewritten_cmd.append(arg)

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{run_dir}:/run",
        "-v", f"{ml_master_root}:/agent",
        "-e", "PATH=/opt/py310/bin:/opt/rl-python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "-e", "PYTHON=/opt/py310/bin/python",
    ]
    if task_dir is not None:
        abs_task_dir = Path(task_dir).resolve()
        docker_cmd += ["-v", f"{abs_task_dir}:/task"]
    if env_file is not None and Path(env_file).exists():
        docker_cmd += ["--env-file", str(env_file)]
    # Pass through selected env vars if present (supports .env-sourced on host)
    included_env: Dict[str, str] = {}
    for key in (env_keys or []):
        val = os.environ.get(key)
        if val:
            docker_cmd += ["-e", f"{key}={val}"]
            included_env[key] = val
    if gpus:
        docker_cmd += ["--gpus", "all"]
    if user:
        docker_cmd += ["--user", user]
    docker_cmd += [image]
    # Execute inside /agent, optionally with a prelude (e.g., start litellm proxy)
    shell_parts: List[str] = ["cd /agent &&"]
    if prelude:
        shell_parts.append(prelude)
        shell_parts.append("&&")
    # Build the command string
    cmd_str = " ".join([json.dumps(c)[1:-1] for c in rewritten_cmd])
    # If run_command_as_user is specified, wrap the command to run as that user
    # This allows prelude to run as root (for pip install, etc.) while command runs as non-root
    if run_command_as_user:
        # Use runuser to switch to the specified user for the command
        # Pass through environment variables with -m (preserve env) or explicitly
        cmd_str = f"runuser -u {run_command_as_user} -- bash -c {json.dumps(cmd_str)}"
    shell_parts.append(cmd_str)
    docker_cmd += ["bash", "-lc", " ".join(shell_parts)]

    return {
        "image": image,
        "mounts": mounts,
        "gpus": gpus,
        "workdir": "/agent",
        "exec": rewritten_cmd,
        "env": included_env,
        "docker_cli": docker_cmd,
        "prelude": prelude,
    }

