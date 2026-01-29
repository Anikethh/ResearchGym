from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure vendored inspect_ai is importable
THIS_DIR = Path(__file__).resolve().parent
INSPECT_SRC = THIS_DIR / "inspect_ai" / "src"
if INSPECT_SRC.exists():
    sys.path.insert(0, str(INSPECT_SRC))

from _async_jobs import cancel_async, check_async, start_async
from _basic_agent_iterative import basic_agent_iterative
from _basic_agent_plus import (
    DEFAULT_CONTINUE_MESSAGE,
    EXTENDED_CONTINUE_MESSAGE,
    basic_agent_plus,
)
from _execute import bash, python
from _file_reader import read_file_chunk, search_file
from _replace import replace_tool
from _write_file import write_file_tool
from _evo_tools import best_candidate_tool, list_candidates_tool, record_candidate_tool
from evo_prompts import DEFAULT_PROMPTS
from templates import additional_notes_template

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.log._file import list_eval_logs, read_eval_log, write_eval_log
from inspect_ai.tool import web_browser, web_search
from utils import get_gpu_generation


CODE_DIR = Path(os.environ.get("CODE_DIR", ".")).resolve()
AGENT_DIR = Path(os.environ.get("AGENT_DIR", ".")).resolve()
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE", str(CODE_DIR))

MAX_TIME_IN_HOURS = os.environ.get("MAX_TIME_IN_HOURS", "0.25")
MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")
DISALLOW_SUBMIT = os.environ.get("DISALLOW_SUBMIT", "false").lower() == "true"
ITERATIVE_AGENT = os.environ.get("ITERATIVE_AGENT", "false").lower() == "true"
USE_EXA_SEARCH = os.environ.get("USE_EXA_SEARCH", "false").lower() == "true"
USE_GOOGLE_WEB_SEARCH = os.environ.get("USE_GOOGLE_WEB_SEARCH", "false").lower() == "true"
USE_EXTENDED_CONTINUE = os.environ.get("RG_EXTENDED_CONTINUE", "false").lower() == "true"
USE_ASYNC = os.environ.get("RG_ENABLE_ASYNC", "false").lower() != "false"
EVO_ENABLED = os.environ.get("RG_EVO_ENABLED", "false").lower() == "true"
USE_IDEA_HINT = os.environ.get("RG_IDEA_HINT", "false").lower() == "true"


def _resolve_browser_enabled() -> bool:
    """Determine whether the browser tool should be exposed to the solver."""
    enable_value = os.environ.get("ENABLE_BROWSER")
    if enable_value is not None:
        return enable_value.strip().lower() == "true"

    disable_value = os.environ.get("DISABLE_BROWSER")
    if disable_value is not None:
        return disable_value.strip().lower() != "true"

    return False


BROWSER_ENABLED = _resolve_browser_enabled()


def _resolve_replace_enabled() -> bool:
    """Enable replace tool by default for non-OpenAI providers, with env overrides."""
    maybe_enable = os.environ.get("RG_ENABLE_REPLACE")
    if maybe_enable is not None:
        return maybe_enable.strip().lower() == "true"

    maybe_disable = os.environ.get("RG_DISABLE_REPLACE")
    if maybe_disable is not None:
        return maybe_disable.strip().lower() != "true"

    provider_prefix = MODEL.split("/", 1)[0].lower()
    return provider_prefix not in {"openai", "azure", "azureai"}


def _resolve_write_enabled() -> bool:
    """Enable write_file tool by default for non-OpenAI providers, with env overrides."""
    maybe_enable = os.environ.get("RG_ENABLE_WRITE_FILE")
    if maybe_enable is not None:
        return maybe_enable.strip().lower() == "true"

    maybe_disable = os.environ.get("RG_DISABLE_WRITE_FILE")
    if maybe_disable is not None:
        return maybe_disable.strip().lower() != "true"

    provider_prefix = MODEL.split("/", 1)[0].lower()
    return provider_prefix not in {"openai", "azure", "azureai"}


def _resolve_log_buffer() -> int | None:
    """Resolve the log buffer threshold for flushing eval logs."""
    raw = os.environ.get("RG_LOG_BUFFER")
    if raw is None or raw.strip() == "":
        return 1
    try:
        parsed = int(raw.strip())
    except ValueError:
        return 1
    return parsed if parsed > 0 else None


REPLACE_ENABLED = _resolve_replace_enabled()
WRITE_FILE_ENABLED = _resolve_write_enabled()
LOG_BUFFER = _resolve_log_buffer()
CONTINUE_MESSAGE = EXTENDED_CONTINUE_MESSAGE if USE_EXTENDED_CONTINUE else DEFAULT_CONTINUE_MESSAGE

gpu_generation = get_gpu_generation()
type_of_processor = gpu_generation if gpu_generation else "CPU"

async_note = ""
if USE_ASYNC:
    async_note = (
        "- For long-running training or evaluation scripts, prefer start_async + check_async/cancel_async and make sure your commands write incremental progress to logs for inspection."
    )

extra_evo_note = ""
if EVO_ENABLED:
    prompt_tags = ", ".join([p.tag for p in DEFAULT_PROMPTS])
    extra_evo_note = (
        "- Evolution mode is enabled. After each grade run, record metrics with record_candidate(metrics_json, note). "
        f"Use prompt variants: {prompt_tags} to diversify mutations."
    )

idea_hint_note = ""
if USE_IDEA_HINT:
    idea_hint_note = "- **Provided idea**: You are given a detailed idea that already works. Implement it faithfully and build on it to further improve performance.\n"

additional_notes = additional_notes_template.substitute(
    type_of_processor=type_of_processor,
    max_time_in_hours=MAX_TIME_IN_HOURS,
    workspace=CODE_DIR,
    workspace_base=WORKSPACE_BASE,
    idea_hint_note=idea_hint_note,
    hypothesis_line="- This is a real research task, the proposed hypotheses should be novel, sound and feasible. You should spell out the details of the method you plan to implement, along with the motivation on why you think it will work.\n" if not USE_IDEA_HINT else "",
    multiple_hypotheses_line="- You can propose multiple hypotheses, run experiments and evaluate them using `grade.py`.\n" if not USE_IDEA_HINT else "",
)

instructions_path = Path(os.environ.get("RG_INSTRUCTIONS_FILE", CODE_DIR / "task_description.md"))
if not instructions_path.exists():
    fallback = CODE_DIR / "instructions.txt"
    if fallback.exists():
        instructions_path = fallback
partial_instructions = instructions_path.read_text() if instructions_path.exists() else ""
async_instructions = f"\n\n{async_note}" if async_note else ""
evo_instruction = f"\n\n{extra_evo_note}" if extra_evo_note else ""
instructions = partial_instructions + additional_notes + async_instructions + evo_instruction


@task
def rg_basic_agent_task():
    if ITERATIVE_AGENT:
        async_tools = [start_async(), check_async(), cancel_async()] if USE_ASYNC else []
        tools = [bash(), read_file_chunk()]
        if REPLACE_ENABLED:
            tools.append(replace_tool())
        if WRITE_FILE_ENABLED:
            tools.append(write_file_tool())
        tools.extend(async_tools)
        if EVO_ENABLED:
            tools += [record_candidate_tool(), list_candidates_tool(), best_candidate_tool()]
        solver = basic_agent_iterative(
            tools=tools,
            max_attempts=1,
            disallow_submit=DISALLOW_SUBMIT,
            real_time_limit=int(float(MAX_TIME_IN_HOURS) * 60 * 60),
            continue_message=CONTINUE_MESSAGE,
        )
    else:
        async_tools = [start_async(), check_async(), cancel_async()] if USE_ASYNC else []
        tool_list = [bash(), python(), read_file_chunk(), search_file()]
        if REPLACE_ENABLED:
            tool_list.append(replace_tool())
        if WRITE_FILE_ENABLED:
            tool_list.append(write_file_tool())
        tool_list.extend(async_tools)
        if EVO_ENABLED:
            tool_list += [record_candidate_tool(), list_candidates_tool(), best_candidate_tool()]
        # Prefer Exa if enabled
        if USE_EXA_SEARCH:
            tool_list.append(web_search(provider="exa", model=MODEL))
        elif USE_GOOGLE_WEB_SEARCH:
            tool_list.append(web_search(provider="google", model=MODEL))
        # Optionally add browser tools
        if BROWSER_ENABLED:
            tool_list += web_browser()

        solver = basic_agent_plus(
            tools=tool_list,
            max_attempts=1,
            disallow_submit=DISALLOW_SUBMIT,
            real_time_limit=int(float(MAX_TIME_IN_HOURS) * 60 * 60),
            max_tool_output=32 * 1024,
            continue_message=CONTINUE_MESSAGE,
        )
    return Task(dataset=[Sample(input=instructions)], solver=solver, sandbox="local")


def run():
    # Use ResearchGym's log directory if available
    log_dir = os.environ.get("RG_LOG_DIR", "./logs")
    budget_limit = float(os.environ.get("RG_BUDGET_LIMIT", "0"))

    print("RGAgentEvolution starting...")
    print(f"Log directory: {log_dir}")
    if budget_limit > 0:
        print(f"Budget limit: ${budget_limit:.2f}")
    else:
        print("Budget limit: None")
    print(f"Model: {MODEL}")
    if EVO_ENABLED:
        print("Evolution DB: enabled (use record_candidate tool after grades)")

    log_path = Path(log_dir).resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    print(f"Created log directory: {log_path}")

    import signal

    def signal_handler(signum, frame):
        print("\nInterrupt received - saving logs...")
        # Let the eval() function handle the interruption gracefully

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    except Exception:
        pass

    logs: list = []
    try:
        model_args: dict[str, object] = {}
        provider_prefix = MODEL.split("/", 1)[0].lower()
        if provider_prefix in {"openai", "azure", "azureai"}:
            model_args["responses_api"] = True

        eval_kwargs: dict[str, object] = {
            "model": MODEL,
            "model_args": model_args,
            "display": "conversation",
            "reasoning_effort": "high",
            "reasoning_tokens": 25600,
            "log_dir": str(log_path),
            "log_format": "eval",
        }
        if LOG_BUFFER is not None:
            eval_kwargs["log_buffer"] = LOG_BUFFER

        logs = eval(rg_basic_agent_task(), **eval_kwargs)
    except KeyboardInterrupt:
        print("Processing interrupted; attempting to recover latest log...")
        try:
            recent_logs = list_eval_logs(str(log_path))
            if recent_logs:
                latest = recent_logs[0]
                candidates = [Path(latest.name), log_path / Path(latest.name).name]
                recovered = False
                for candidate in candidates:
                    try:
                        logs = [read_eval_log(str(candidate))]
                        print(f"Recovered log from {candidate}")
                        recovered = True
                        break
                    except Exception:
                        continue
                if not recovered:
                    logs = []
            else:
                logs = []
        except Exception as exc:
            print(f"Log recovery failed: {exc}")
            logs = []
    except Exception:
        raise

    # Write a secondary log in the alternate format alongside the primary
    for log in logs:
        try:
            if not getattr(log, "location", None):
                continue
            base, ext = os.path.splitext(log.location)
            alt_ext = ".eval" if ext == ".json" else ".json"
            alt_path = f"{base}{alt_ext}"
            write_eval_log(log, location=alt_path, format="auto")
            print(f"Secondary log written: {alt_path}")
        except Exception as e:
            print(f"Secondary log write failed: {e}")

    # Also maintain a global ResearchGym/logs dir with only .eval files
    try:
        researchgym_root = THIS_DIR.parent.parent.resolve()
        global_logs_dir = researchgym_root / "logs"
        global_logs_dir.mkdir(parents=True, exist_ok=True)

        for log in logs:
            try:
                if not getattr(log, "location", None):
                    continue
                base_name = os.path.splitext(os.path.basename(log.location))[0]
                global_eval_path = global_logs_dir / f"{base_name}.eval"
                write_eval_log(log, location=str(global_eval_path), format="eval")
                print(f"Copied .eval to global logs: {global_eval_path}")
            except Exception as e:
                print(f"Global .eval copy failed: {e}")
    except Exception as e:
        print(f"Global logs setup failed: {e}")


if __name__ == "__main__":
    run()
