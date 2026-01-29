from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from inspect_ai.tool import Tool, tool

from _execute import get_shell_command

DONE_STATES = {"completed", "failed", "cancelled"}


def _display_path(path: Path) -> str:
    """Return a redacted path for agent-facing responses."""
    code_dir = os.environ.get("CODE_DIR")
    workspace = os.environ.get("WORKSPACE_BASE")
    if code_dir:
        try:
            return str(path.resolve().relative_to(Path(code_dir).resolve()))
        except Exception:
            pass
    if workspace:
        try:
            return str(path.resolve().relative_to(Path(workspace).resolve()))
        except Exception:
            pass
    return path.name


def _timestamp() -> str:
    """Return a UTC timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jobs_root() -> Path:
    """Return the root directory for async jobs."""
    code_dir = Path(os.environ.get("CODE_DIR", ".")).resolve()
    workspace_base = os.environ.get("WORKSPACE_BASE")
    default_root = code_dir / "async_jobs"
    env_dir = os.environ.get("RG_ASYNC_JOBS_DIR")
    if env_dir:
        try:
            candidate = Path(env_dir).resolve()
            candidate.relative_to(code_dir)
            return candidate
        except Exception:
            # If the override escapes CODE_DIR, ignore it to keep the agent sandboxed.
            _log_debug(f"{_timestamp()} ignored RG_ASYNC_JOBS_DIR outside CODE_DIR: {env_dir}")

    if workspace_base:
        try:
            default_root.relative_to(Path(workspace_base).resolve())
        except Exception:
            # Fall back to a safe location inside the workspace if CODE_DIR is misconfigured.
            return Path(workspace_base).resolve() / "input" / "async_jobs"

    return default_root


def _log_debug(msg: str) -> None:
    """Write lightweight debug info for async events without raising."""
    try:
        root = _jobs_root()
        root.mkdir(parents=True, exist_ok=True)
        log = root / "async_debug.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")
    except Exception:
        try:
            print(msg)
        except Exception:
            pass


def _resolve_workdir(workdir: Optional[str]) -> Path:
    """Resolve the working directory for a job, defaulting to CODE_DIR."""
    code_dir = Path(os.environ.get("CODE_DIR", ".")).resolve()
    workspace_base = os.environ.get("WORKSPACE_BASE")
    candidate = code_dir if not workdir else Path(workdir)
    if not candidate.is_absolute():
        candidate = (code_dir / candidate).resolve()
    if workspace_base:
        base_resolved = Path(workspace_base).resolve()
        try:
            candidate.relative_to(base_resolved)
        except Exception:
            raise ValueError(f"workdir must stay within WORKSPACE_BASE: {base_resolved}")
    if not candidate.exists():
        raise FileNotFoundError(f"workdir does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"workdir is not a directory: {candidate}")
    return candidate


def _base_env() -> Dict[str, str]:
    """Build a minimal, safe environment for subprocesses."""
    env = os.environ.copy()
    venv_bin = os.path.dirname(sys.executable)
    env.update(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PATH": f"{venv_bin}{os.pathsep}" + env.get("PATH", ""),
        }
    )
    return env


def _tail_file(path: Path, lines: int = 80) -> str:
    """Return the last `lines` lines of the given file."""
    if not path.exists():
        return ""
    dq: deque[str] = deque(maxlen=max(lines, 1))
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            dq.append(line.rstrip("\n"))
    return "\n".join(dq)


def _pid_alive(pid: int) -> bool:
    """Check if a PID is currently alive."""
    try:
        if pid <= 0:
            return False
        # SIG_DFL check; signal 0 is non-destructive
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # We can't signal it, assume alive to avoid false negatives
        return True
    except Exception:
        return False


def _write_metadata(meta_path: Path, data: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _watch_process(proc: subprocess.Popen, meta_path: Path) -> None:
    """Wait for process completion and persist final status."""
    try:
        rc = proc.wait()
        status = "completed" if rc == 0 else "failed"
        finish = _timestamp()
        meta = _load_metadata(meta_path)
        # Preserve explicit cancellations
        if meta.get("status") in DONE_STATES and meta.get("status") != "running":
            pass
        else:
            meta["status"] = status
        meta["returncode"] = rc
        meta["ended_at"] = finish
        _write_metadata(meta_path, meta)
    except Exception:
        # Best-effort; avoid crashing the agent
        return


def _start_watcher(proc: subprocess.Popen, meta_path: Path) -> None:
    thread = threading.Thread(target=_watch_process, args=(proc, meta_path), daemon=True)
    thread.start()


def _terminate_process(meta: Dict[str, Any]) -> bool:
    """Request termination of a running process."""
    pid = meta.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    system = platform.system()
    try:
        if system != "Windows":
            pgid = meta.get("pgid")
            if isinstance(pgid, int) and pgid > 0:
                os.killpg(pgid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            time.sleep(0.3)
            if _pid_alive(pid):
                try:
                    if isinstance(pgid, int) and pgid > 0:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except Exception:
        return False


@tool
def start_async() -> Tool:
    """Start a long-running shell command without blocking. Returns a job_id you can poll later."""

    async def execute(cmd: str, workdir: Optional[str] = None) -> str:
        """
        Launch a background shell command and return a job handle immediately.

        Args:
          cmd (str): Command to run (passed to the shell).
          workdir (Optional[str]): Working directory for the command; defaults to CODE_DIR. Must stay inside CODE_DIR/WORKSPACE_BASE.
        """
        if not cmd or not cmd.strip():
            return "error: command is required"
        try:
            cwd = _resolve_workdir(workdir)
        except Exception as exc:
            return f"error: {exc}"

        jobs_dir = _jobs_root()
        job_id = uuid4().hex
        job_dir = jobs_dir / job_id
        log_path = job_dir / "stdout_stderr.log"
        meta_path = job_dir / "metadata.json"
        _log_debug(f"[start_async] job_id={job_id} cmd={cmd} workdir={cwd}")

        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as handle:
                # Isolate the child from our console so stray CTRL+C/CTRL+BREAK events
                # do not bubble in as KeyboardInterrupt. Cancellation should only happen
                # via cancel_async().
                creationflags = 0
                if platform.system() == "Windows":
                    creationflags = (
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        | getattr(subprocess, "DETACHED_PROCESS", 0)
                        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    )
                proc = subprocess.Popen(
                    get_shell_command() + [cmd],
                    cwd=str(cwd),
                    env=_base_env(),
                    stdout=handle,
                    stderr=handle,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    creationflags=creationflags,
                )
        except Exception as exc:
            return f"error: failed to start process: {exc}"

        pgid: Optional[int] = None
        if platform.system() != "Windows":
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = None

        metadata: Dict[str, Any] = {
            "job_id": job_id,
            "command": cmd,
            "cwd": str(cwd),
            "pid": proc.pid,
            "pgid": pgid,
            "status": "running",
            "log_path": str(log_path),
            "started_at": _timestamp(),
        }
        _write_metadata(meta_path, metadata)
        _start_watcher(proc, meta_path)

        response = {
            "job_id": job_id,
            "status": "running",
            "pid": proc.pid,
            "log_path": _display_path(log_path),
            "cwd": _display_path(cwd),
        }
        _log_debug(f"[start_async] started job_id={job_id} pid={proc.pid} log={log_path}")
        return json.dumps(response, indent=2)

    return execute


@tool
def check_async() -> Tool:
    """Check the status of a background command and optionally return a log tail."""

    async def execute(job_id: str, tail_lines: int = 80, sleep_minutes: float = 0.0) -> str:
        """
        Inspect a background job's status and (optionally) delay before polling.

        Args:
          job_id (str): Identifier returned by start_async.
          tail_lines (int): Number of log lines to include from the end of the job log.
          sleep_minutes (float): Optional delay before polling, expressed in minutes.
        """
        if sleep_minutes and sleep_minutes > 0:
            # Cooperative delay so the model can throttle polling
            try:
                time.sleep(max(sleep_minutes * 60, 0))
            except Exception:
                pass
        _log_debug(f"[check_async] job_id={job_id} tail_lines={tail_lines} sleep_minutes={sleep_minutes}")

        jobs_dir = _jobs_root()
        meta_path = jobs_dir / job_id / "metadata.json"
        metadata = _load_metadata(meta_path)
        if not metadata:
            return f"error: job {job_id} not found"

        status = metadata.get("status", "unknown")
        pid = metadata.get("pid")
        if status == "running" and isinstance(pid, int):
            if not _pid_alive(pid):
                # Process ended but watcher may not have flushed yet
                metadata["status"] = "unknown"
                try:
                    _write_metadata(meta_path, metadata)
                except Exception:
                    pass
                status = metadata["status"]

        log_path = Path(metadata.get("log_path", jobs_dir / job_id / "stdout_stderr.log"))
        tail = _tail_file(log_path, lines=tail_lines)

        response: Dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "pid": pid,
            "returncode": metadata.get("returncode"),
            "log_path": _display_path(log_path),
            "tail": tail,
        }
        _log_debug(f"[check_async] job_id={job_id} status={status} pid={pid} returncode={metadata.get('returncode')}")
        return json.dumps(response, indent=2)

    return execute


@tool
def cancel_async() -> Tool:
    """Request cancellation of a running background command."""

    async def execute(job_id: str) -> str:
        """
        Send a termination signal to a running background job.

        Args:
          job_id (str): Identifier returned by start_async.
        """
        jobs_dir = _jobs_root()
        meta_path = jobs_dir / job_id / "metadata.json"
        metadata = _load_metadata(meta_path)
        if not metadata:
            return f"error: job {job_id} not found"
        _log_debug(f"[cancel_async] job_id={job_id} status={metadata.get('status')} pid={metadata.get('pid')}")

        status = metadata.get("status", "unknown")
        if status in DONE_STATES:
            return json.dumps(
                {
                    "job_id": job_id,
                    "status": status,
                    "returncode": metadata.get("returncode"),
                    "log_path": metadata.get("log_path"),
                    "message": "job already finished",
                },
                indent=2,
            )

        terminated = _terminate_process(metadata)
        metadata["status"] = "cancelled" if terminated else status
        metadata["ended_at"] = _timestamp()
        _write_metadata(meta_path, metadata)

        response = {
            "job_id": job_id,
            "status": metadata["status"],
            "log_path": _display_path(Path(metadata.get("log_path", jobs_dir / job_id / "stdout_stderr.log"))),
            "message": "termination signal sent" if terminated else "failed to signal process",
        }
        _log_debug(f"[cancel_async] job_id={job_id} terminated={terminated}")
        return json.dumps(response, indent=2)

    return execute
