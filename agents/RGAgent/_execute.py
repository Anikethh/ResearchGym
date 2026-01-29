from __future__ import annotations

from inspect_ai.tool import Tool, tool
from inspect_ai.tool._tool_call import ToolCall, ToolCallContent, ToolCallView, ToolCallViewer
from inspect_ai.util import sandbox
import os
import platform
import re
import sys
from pathlib import Path


def get_shell_command() -> list[str]:
    """Get the appropriate shell command for the current platform.

    On Windows, prefer a real Bash if available (e.g., from Git for Windows),
    falling back to cmd if not found. This preserves the tool name "bash"
    without requiring WSL/Docker.
    """
    if platform.system() == "Windows":
        import shutil
        # Allow explicit override
        bash_override = os.environ.get("RG_BASH_PATH")
        if bash_override and os.path.exists(bash_override):
            return [bash_override, "--login", "-c"]

        # Prefer a real Git Bash if present
        git_candidates = [
            shutil.which("bash") or "",
            r"C:\\Program Files\\Git\\bin\\bash.exe",
            r"C:\\Program Files\\Git\\usr\\bin\\bash.exe",
        ]
        for c in git_candidates:
            if not c:
                continue
            # Skip WSL stubs from WindowsApps/System32
            lower = c.lower()
            if ("windowsapps\\bash.exe" in lower) or ("system32\\bash.exe" in lower) or ("sysnative\\bash.exe" in lower):
                continue
            if os.path.exists(c):
                return [c, "--login", "-c"]
        # Fallback to cmd
        return ["cmd", "/c"]
    else:
        # Use bash on Unix-like systems
        return ["bash", "--login", "-c"]


def code_viewer(language: str, code_param: str) -> ToolCallViewer:
    def viewer(tool_call: ToolCall) -> ToolCallView:
        code = tool_call.arguments.get(code_param, None)
        try:
            code = (code or tool_call.function).strip()
        except Exception as e:
            code = f"ERROR: unable to parse tool call: {e}"
        call = ToolCallContent(
            title=language,
            format="markdown",
            content=f"```{language}\n" + code + "\n```\n",
        )
        return ToolCallView(call=call)

    return viewer

@tool(viewer=code_viewer("bash", "cmd"))
def bash(timeout: int | None = None, user: str | None = None) -> Tool:
    """Bash shell command execution tool.

    Execute bash shell commands using a sandbox environment (e.g. "docker").

    Args:
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(cmd: str) -> str:
        """
        Use this function to execute bash commands.

        Args:
          cmd (str): The bash command to execute.

        Returns:
          The output of the command.
        """
        # execute the command
        try:
            original_cmd = cmd
            # normalize common alias 'applypatch' -> 'apply_patch'
            try:
                cmd = re.sub(r'(?<![\w/\./-])applypatch(?![\w/\./-])', 'apply_patch', cmd)
            except Exception:
                pass
            # Special handling for apply_patch here-doc: extract the patch body and pipe
            # it to apply_patch.py via stdin to avoid fragile shell here-doc parsing.
            try:
                if ("apply_patch" in cmd) and ("<<" in cmd):
                    # Find the here-doc tag, e.g., << 'PATCH' or <<PATCH
                    m = re.search(r"<<\s*(['\"])?(?P<tag>[A-Za-z0-9_]+)\1", cmd)
                    if m:
                        tag = m.group('tag')
                        # Start of content is the newline after the heredoc opener
                        content_start = cmd.find("\n", m.end())
                        if content_start != -1:
                            # Search for terminator on its own line
                            term_re = re.compile(rf"^[ \t]*{re.escape(tag)}[ \t]*$", re.M)
                            term_match = term_re.search(cmd, pos=content_start + 1)
                            patch_end = term_match.start() if term_match else len(cmd)
                            patch_body = cmd[content_start + 1 : patch_end]

                            # Compute Python and script paths
                            py = sys.executable
                            agent_dir = os.environ.get("AGENT_DIR")
                            if agent_dir:
                                script = str(Path(agent_dir) / "apply_patch.py")
                            else:
                                script = str(Path(__file__).resolve().with_name("apply_patch.py"))
                            # Prefer forward slashes for Git Bash on Windows
                            try:
                                is_windows = platform.system() == "Windows"
                                shell = get_shell_command()
                                using_bash = any("bash" in (s or "").lower() for s in shell)
                                if is_windows and using_bash:
                                    py = py.replace("\\", "/")
                                    script = script.replace("\\", "/")
                            except Exception:
                                pass

                            # Default working directory for tools is CODE_DIR (task input workspace)
                            code_dir = os.environ.get("CODE_DIR")
                            cwd = os.path.abspath(code_dir) if code_dir else None

                            venv_bin = os.path.dirname(sys.executable)
                            env = os.environ.copy()
                            env.update({
                                "DEBIAN_FRONTEND": "noninteractive",
                                "GIT_TERMINAL_PROMPT": "0",
                                "PATH": f"{venv_bin}{os.pathsep}" + env.get("PATH", ""),
                            })

                            result = await sandbox().exec(
                                cmd=[py, script],
                                input=patch_body,
                                timeout=30 if "<<" in original_cmd else None,
                                user=user,
                                env=env,
                                cwd=cwd,
                            )
                            output = ""
                            if result.stderr:
                                output = f"{result.stderr}\n"
                            return f"{output}{result.stdout}"
            except Exception:
                # Fall back to shell execution below
                pass
            # If the model invokes apply_patch, rewrite it to call the bundled
            # Python script directly so Git Bash on Windows doesn't need an
            # external apply_patch binary on PATH. Preserve here-doc syntax.
            try:
                if "apply_patch" in cmd:
                    # Compute Python and script paths
                    py = sys.executable
                    agent_dir = os.environ.get("AGENT_DIR")
                    if agent_dir:
                        script = str(Path(agent_dir) / "apply_patch.py")
                    else:
                        script = str(Path(__file__).resolve().with_name("apply_patch.py"))
                    # When running under Git Bash on Windows, prefer forward slashes
                    # to avoid backslash escape/translation issues.
                    try:
                        is_windows = platform.system() == "Windows"
                        shell = get_shell_command()
                        using_bash = any("bash" in (s or "").lower() for s in shell)
                        if is_windows and using_bash:
                            py = py.replace("\\", "/")
                            script = script.replace("\\", "/")
                    except Exception:
                        pass
                    repl = f'"{py}" "{script}"'
                    # Only rewrite before the first here-doc delimiter if present
                    split_idx = cmd.find("<<")
                    if split_idx != -1:
                        head, tail = cmd[:split_idx], cmd[split_idx:]
                    else:
                        head, tail = cmd, ""
                    # Replace the first standalone occurrence of apply_patch
                    head_new = re.sub(r'(?<![\w/\./-])apply_patch(?![\w/\./-])', repl, head, count=1)
                    cmd = head_new + tail
            except Exception:
                pass
            # stop hanging on stdin for apply_patch
            cmd_specific_timeout = None
            if "apply_patch" in cmd and "<<" in cmd:
                cmd_specific_timeout = 30

            # Default working directory for tools is CODE_DIR (task input workspace)
            code_dir = os.environ.get("CODE_DIR")
            cwd = os.path.abspath(code_dir) if code_dir else None

            # ensure child processes prefer the current venv's bin on PATH
            venv_bin = os.path.dirname(sys.executable)
            env = os.environ.copy()
            env.update({
                "DEBIAN_FRONTEND": "noninteractive",
                "GIT_TERMINAL_PROMPT": "0",
                "PATH": f"{venv_bin}{os.pathsep}" + env.get("PATH", ""),
            })

            # Normalize line endings for Git Bash on Windows to avoid here-doc
            # terminator mismatches (e.g., 'PATCH\r' vs 'PATCH').
            try:
                if platform.system() == "Windows":
                    shell = get_shell_command()
                    using_bash = any("bash" in (s or "").lower() for s in shell)
                    if using_bash and "\r\n" in cmd:
                        cmd = cmd.replace("\r\n", "\n")
            except Exception:
                pass

            result = await sandbox().exec(
                cmd=get_shell_command() + [cmd],
                timeout=cmd_specific_timeout,
                user=user,
                env=env,
                cwd=cwd,
            )
            # return output (including stderr if any)
            output = ""
            if result.stderr:
                output = f"{result.stderr}\n"
            return f"{output}{result.stdout}"
        except Exception as e:
            return f"ERROR: unable to execute command: {e}"

    return execute


@tool(viewer=code_viewer("python", "code"))
def python(timeout: int | None = None, user: str | None = None) -> Tool:
    """Python code execution tool.

    Execute Python code using a sandbox environment (e.g. "docker").

    Args:
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(code: str) -> str:
        """
        Use the python function to execute Python code.

        The python function will only return you the stdout of the script,
        so make sure to use print to see the output.

        Args:
          code (str): The python code to execute.

        Returns:
          The output of the Python code.
        """
        # Default working directory for tools is CODE_DIR (task input workspace)
        code_dir = os.environ.get("CODE_DIR")
        cwd = os.path.abspath(code_dir) if code_dir else None

        result = await sandbox().exec(
            cmd=[sys.executable],
            input=code,
            timeout=timeout,
            user=user,
            cwd=cwd,
        )
        # return output (including stderr if any)
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


