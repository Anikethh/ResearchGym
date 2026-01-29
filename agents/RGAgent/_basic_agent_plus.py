from __future__ import annotations

# Minimal vendored copy of BasicAgent (plus) customized for ResearchGym.

import asyncio
import os
import time
from datetime import datetime, timezone
from json import JSONDecodeError
from logging import getLogger
from textwrap import dedent
from typing import Callable, cast

import json
from pathlib import Path

import tiktoken
from inspect_ai._util.format import format_progress_time
from inspect_ai.model._cache import CachePolicy
from inspect_ai.model._call_tools import call_tools
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import get_model, model_usage as inspect_model_usage
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.scorer._metric import Score, ValueToFloat, value_to_float
from inspect_ai.scorer._score import score
from inspect_ai.solver._chain import chain
from inspect_ai.solver._prompt import system_message
from inspect_ai.solver._solver import Generate, Solver, solver
from inspect_ai.solver._task_state import TaskState
from inspect_ai.solver._use_tools import use_tools
from inspect_ai.tool._tool import Tool, ToolResult, tool
from inspect_ai.tool._tool_call import ToolCallError
from inspect_ai.tool._tool_with import tool_with
from openai import LengthFinishReasonError
from typing_extensions import TypedDict, Unpack
from tenacity import RetryError
from utils import generate_patched, prune_messages

logger = getLogger(__name__)
RUN_STARTED_AT_ISO = datetime.now(timezone.utc).isoformat()
_RUN_STARTED_MONO = time.time()


def _tool_timeout_messages(tool_calls: list | None) -> list[ChatMessageTool]:
    """Emit synthetic tool outputs when a call times out so history stays valid."""
    messages: list[ChatMessageTool] = []
    if not tool_calls:
        return messages
    for tool_call in tool_calls:
        call_id = getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", None)
        function_name = getattr(tool_call, "function", None)
        messages.append(
            ChatMessageTool(
                content="Tool execution timed out before producing output.",
                tool_call_id=call_id,
                function=function_name,
                error=ToolCallError(
                    type="timeout",
                    message="Tool execution exceeded the allotted time limit before returning.",
                ),
            )
        )
    return messages


def _prune_incomplete_tool_calls(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Remove trailing assistant tool calls with no recorded outputs (e.g., abrupt stop)."""
    pending: set[str] = set()
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls or []:
                call_id = getattr(call, "id", None)
                if call_id:
                    pending.add(call_id)
        elif isinstance(msg, ChatMessageTool) and msg.tool_call_id:
            pending.discard(msg.tool_call_id)

    if not pending:
        return messages

    pruned = list(messages)
    removed = 0
    while pending and pruned:
        msg = pruned.pop()
        removed += 1
        if isinstance(msg, ChatMessageAssistant) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls or []:
                call_id = getattr(call, "id", None)
                if call_id:
                    pending.discard(call_id)
        elif isinstance(msg, ChatMessageTool) and msg.tool_call_id:
            pending.discard(msg.tool_call_id)
    if removed:
        logger.warning(
            "Resume cleanup removed %d trailing messages without matching tool outputs.",
            removed,
        )
    return pruned

HANDOFF_TOKEN_THRESHOLD_DEFAULT = 140_000
HANDOFF_SUMMARY_PROMPT = dedent(
    """
    You have exceeded the maximum number of tokens, please stop coding and instead write a short memento message for yourself. Your note should:

    - Summarize what you finished and what still needs work.
    - List out your current understanding of the repository and highlight the files that are central to implementing a new method.
    - Summarize your proposed approach and point to it in the codebase.
    - Note your observations from any experiments you have run so far.
    - Clarify what your next steps would be if you had more time, any open issues in current implementation.

    Do not call tools or run code; respond with plain text only.
    """
).strip()

HANDOFF_BRIDGE_PROMPT = dedent(
    """
    You were originally given instructions from a user about the research task. Here were the user messages:

    {user_messages_text}

    You attempted to solve this problem and produced a summary of your work. Here is the summary, leverage this information and continue your work for improving performance on the original task:

    {summary_text}

    Continue the work from here. You can choose to extend this method or propose alternate ideas based on observation and insights from current progress.
    """
).strip()

HANDOFF_MESSAGE_OVERHEAD_TOKENS = 6


def _resolve_conversation_tokenizer(model_name: str) -> tiktoken.Encoding:
    """Resolve a tokenizer suitable for estimating chat context length."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        normalized = model_name.lower()
        if "gpt-4" in normalized or "gpt-5" in normalized or "o1" in normalized or "o4" in normalized:
            base_encoding = "o200k_base"
        elif "gpt-3.5" in normalized or "claude" in normalized:
            base_encoding = "cl100k_base"
        else:
            base_encoding = "o200k_base"
        return tiktoken.get_encoding(base_encoding)


def _message_text(msg: ChatMessage) -> str:
    """Extract message text for token estimation."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts: list[str] = []
        for item in msg.content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                parts.append(getattr(item, "text", "") or "")
            elif item_type == "reasoning":
                parts.append(getattr(item, "reasoning", "") or "")
        return "\n".join(filter(None, parts))
    return getattr(msg, "text", "") or ""


def estimate_conversation_tokens(
    messages: list[ChatMessage],
    tokenizer: tiktoken.Encoding,
    per_message_overhead: int = HANDOFF_MESSAGE_OVERHEAD_TOKENS,
) -> int:
    """Approximate the live context length of the conversation in tokens."""
    total_tokens = 0
    for message in messages:
        text = _message_text(message)
        if not text:
            continue
        try:
            total_tokens += len(tokenizer.encode(text, disallowed_special=()))
        except Exception:
            total_tokens += max(1, len(text) // 4)
    total_tokens += per_message_overhead * len(messages)
    return total_tokens


SYSTEM_MESSAGE_BASE = """
You are an agent tasked with proposing and testing your novel scientific ideas on a given research goal. You have several functions available to help with this. Each message may
may perform one function call. You will see the result of the function right
after sending the message. If you need to perform multiple actions, you can
always send more messages with subsequent function calls. Do some reasoning
before your actions, describing what function calls you are going to use and
how they fit into your plan.

Use as much of the available time as possible to refine and optimize your solution: Your goal is to achieve the best possible score, not just to submit your first working solution. Keep working until the time limit expires.

YOU MUST use the bash and python tools provided provided to implement your ideas. 

Only execute a single command using each tool at once.

This is a multi-step task. DO THINGS BIT BY BIT by slowly building up your solution with multiple tool calls, don't try to do everything in one go. Just reason about what is the best next step and execute it. Then take the next step after you receive the output of the previous step.

Do not stop until you have produced multiple results using your ideas. Before calling end_task, YOU MUST have an entire codebase that can produce results on the required evaluation metrics."""

GRADE_MESSAGE = """
Ensure that running grade.py will produce results on all the result tables! You can check grade.py to ensure that it faithfully tests your method on the evaluation to help with your development. You can use this multiple times, iteratively learn from the results and improve your method."""

SYSTEM_MESSAGE_END = """
Remember, you are tasked with the task of proposing and testing your novel ideas on the given research goal."""

DEFAULT_SYSTEM_MESSAGE = SYSTEM_MESSAGE_BASE + GRADE_MESSAGE + SYSTEM_MESSAGE_END

DEFAULT_INCORRECT_MESSAGE = """
Your submission was incorrect. Please proceed and attempt to find the correct answer.
"""
DEFAULT_CONTINUE_MESSAGE = "Please proceed to the next step using your best judgement. If you believe you are finished, double check your work to continue to refine and improve your submission."
EXTENDED_CONTINUE_MESSAGE = "Please proceed to the next step using your best judgement. You should always strive to improve performance further, if you have ideas on how to improve the current " \
"proposed solution, you can do that. Otherwise, if you feel the current solution is sub-optimal, feel free to propose completely new ideas to improve performance further. If you arec absolutely sure " \
"that you have reached the best possible performance, and verified with results on ALL sub-tasks, only then you can choose to end the task. You should be ambitious, improve the performance and keep continuing even if you beat the baseline."
DEFAULT_SUBMIT_NAME = "end_task"
DEFAULT_SUBMIT_DESCRIPTION = (
    "Signal that you are completely finished and that you would like to end the task."
)


class BasicAgentDeprecatedArgs(TypedDict, total=False):
    max_messages: int | None


@solver
def basic_agent_plus(
    *,
    init: Solver | list[Solver] | None = None,
    tools: list[Tool] | Solver | None = None,
    cache: bool | CachePolicy = False,
    max_attempts: int = 1,
    message_limit: int | None = None,
    token_limit: int | None = None,
    real_time_limit: int | None = None,
    max_tool_output: int | None = None,
    score_value: ValueToFloat | None = None,
    incorrect_message: str | Callable[[TaskState, list[Score]], str] = DEFAULT_INCORRECT_MESSAGE,
    continue_message: str = DEFAULT_CONTINUE_MESSAGE,
    submit_name: str = DEFAULT_SUBMIT_NAME,
    submit_description: str = DEFAULT_SUBMIT_DESCRIPTION,
    disallow_submit: bool = False,
    **kwargs: Unpack[BasicAgentDeprecatedArgs],
) -> Solver:
    """Basic ReAct agent.

    Agent that runs a tool use loop. Tailor the model's instructions by passing a `system_message()` and/or other steps to `init` (if no `init` is specified then a default system
    message will be used). Use `max_attempts` to support additional submissions if
    the initial submission(s) are incorrect.

    Submissions are evaluated using the task's main scorer, with value of 1.0
    indicating a correct answer. Scorer values are converted to float (e.g.
    "C" becomes 1.0) using the standard value_to_float() function. Provide an
    alternate conversion scheme as required via `score_value`.

    Args:
       init: (Solver | list[Solver] | None): Agent initialisation
         (defaults to system_message with basic ReAct prompt)
       tools (list[Tool | ToolDef] | Solver | None): Tools available for the agent. Either a
         list of tools or a Solver that can yield dynamic tools per-sample.
       cache: (bool | CachePolicy): Caching behaviour for generate responses
         (defaults to no caching).
       max_attempts (int): Maximum number of submissions to accept before terminating.
       message_limit (int | None): Limit on messages in sample before terminating agent.
          If not specified, will use limit_messages defined for the task. If there is none
          defined for the task, 50 will be used as a default.
       token_limit (int | None): Limit on tokens used in sample before terminating agent.
       max_tool_output (int | None): Maximum output length (in bytes).
          Defaults to max_tool_output from active GenerateConfig.
       score_value (ValueToFloat): Function used to extract float from scores (defaults
         to standard value_to_float())
       incorrect_message (str | Callable[[TaskState, list[Score]], str]): User message reply for an
         incorrect submission from the model. Alternatively, a function which returns a message.
       continue_message (str): User message to urge the model to continue when it
         doesn't make a tool call.
       **kwargs (Any): Deprecated arguments for backward compatibility.

    Returns:
        Plan for agent.
    """
    # resolve deprecated
    for arg, value in kwargs.items():
        if arg == "max_messages":
            # deprecated, don't warn yet
            message_limit = int(cast(int, value))

    # resolve init
    if init is None:
        sys_message = DEFAULT_SYSTEM_MESSAGE
        init = system_message(sys_message)
    init = init if isinstance(init, list) else [init]

    # resolve tools
    if tools is None:
        tools = []
    tools = tools if isinstance(tools, Solver) else use_tools(tools)

    # resolve score_value function
    score_value_fn = score_value or value_to_float()

    # submission tool
    @tool
    def end_task() -> Tool:
        async def execute(end_message: str) -> ToolResult:
            """Signal that you are completely finished.

            Args:
              end_message (str): Final message to signal that you are finished.
            """
            return end_message

        return execute

    # solver that adds submission tool
    @solver
    def end_task_tool() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.tools.append(tool_with(end_task(), submit_name, submit_description))
            return state

        return solve

    # helper to extract a submitted answer
    def submission(tool_results: list[ChatMessageTool]) -> str | None:
        return next(
            (result.text for result in tool_results if result.function == submit_name),
            None,
        )

    # main agent loop
    @solver
    def basic_agent_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # resolve message_limit -- prefer parameter then fall back to task
            state.message_limit = message_limit or state.message_limit

            # resolve token limit
            state.token_limit = token_limit or state.token_limit

            # track attempts
            attempts = 0

            # Configure token handoff behaviour (optional override via env vars)
            summary_prompt = os.environ.get(
                "RG_TOKEN_HANDOFF_SUMMARY_PROMPT", HANDOFF_SUMMARY_PROMPT
            )
            bridge_prompt_template = os.environ.get(
                "RG_TOKEN_HANDOFF_BRIDGE_PROMPT", HANDOFF_BRIDGE_PROMPT
            )

            raw_threshold = os.environ.get("RG_TOKEN_HANDOFF_THRESHOLD", "").strip()
            handoff_threshold: int | None
            if raw_threshold:
                try:
                    handoff_threshold = int(raw_threshold)
                except ValueError:
                    logger.warning(
                        "Invalid RG_TOKEN_HANDOFF_THRESHOLD=%s; defaulting to %d",
                        raw_threshold,
                        HANDOFF_TOKEN_THRESHOLD_DEFAULT,
                    )
                    handoff_threshold = HANDOFF_TOKEN_THRESHOLD_DEFAULT
            else:
                handoff_threshold = HANDOFF_TOKEN_THRESHOLD_DEFAULT

            if handoff_threshold <= 0:
                logger.info(
                    "Token handoff disabled because threshold %s is not positive",
                    handoff_threshold,
                )
                handoff_threshold = None

            raw_max_handoffs = os.environ.get("RG_TOKEN_HANDOFF_MAX", "").strip()
            max_handoffs: int | None = None
            if raw_max_handoffs:
                try:
                    parsed_max = int(raw_max_handoffs)
                    if parsed_max > 0:
                        max_handoffs = parsed_max
                    else:
                        logger.info(
                            "Ignoring RG_TOKEN_HANDOFF_MAX=%s because it is not positive",
                            raw_max_handoffs,
                        )
                except ValueError:
                    logger.warning(
                        "Invalid RG_TOKEN_HANDOFF_MAX=%s; allowing unlimited handoffs",
                        raw_max_handoffs,
                    )

            # Capture initial system + user instructions so we can rebuild context later
            initial_system_messages = [
                msg.model_copy(deep=True)
                for msg in state.messages
                if isinstance(msg, ChatMessageSystem)
            ]
            initial_user_messages = [
                msg.model_copy(deep=True)
                for msg in state.messages
                if isinstance(msg, ChatMessageUser)
            ]

            initial_user_text_parts: list[str] = []
            for msg in initial_user_messages:
                text = msg.text.strip()
                if text:
                    initial_user_text_parts.append(text)
            original_user_messages_text = "\n\n".join(initial_user_text_parts)

            if not original_user_messages_text:
                sample_input = state.input
                if isinstance(sample_input, str):
                    original_user_messages_text = sample_input.strip()
                elif isinstance(sample_input, list):
                    input_parts: list[str] = []
                    for msg in sample_input:
                        if isinstance(msg, ChatMessageUser):
                            text = msg.text.strip()
                            if text:
                                input_parts.append(text)
                    original_user_messages_text = "\n\n".join(input_parts)

            if not original_user_messages_text:
                original_user_messages_text = "(no original user instructions available)"

            handoff_history: list[str] = []
            handoff_count = 0
            next_handoff_token = handoff_threshold if handoff_threshold else None
            handoff_active = False
            conversation_tokenizer: tiktoken.Encoding | None = None
            tokenizer_model_name_str = ""

            def perform_handoff(summary_text: str, *, context_tokens: int | None = None) -> None:
                """Reset the conversation with a fresh context window and carry over state."""
                nonlocal handoff_count, handoff_active, next_handoff_token

                cleaned_summary = summary_text.strip() if summary_text else ""
                if not cleaned_summary:
                    cleaned_summary = (
                        "Summary unavailable: the prior agent did not provide any details."
                    )

                handoff_history.append(cleaned_summary)
                combined_summary = "\n\n---\n\n".join(
                    f"Handoff {idx + 1} Summary:\n{summary}"
                    for idx, summary in enumerate(handoff_history)
                )
                bridge_prompt = bridge_prompt_template.format(
                    user_messages_text=original_user_messages_text,
                    summary_text=combined_summary,
                )

                tokenizer = conversation_tokenizer or _resolve_conversation_tokenizer(
                    tokenizer_model_name_str
                )
                if context_tokens is None:
                    context_tokens = estimate_conversation_tokens(state.messages, tokenizer)
                logger.info(
                    "Resetting conversation after token handoff #%s (context_tokens=%s).",
                    handoff_count + 1,
                    context_tokens,
                )

                new_messages: list[ChatMessage] = []
                if initial_system_messages:
                    new_messages.extend(
                        msg.model_copy(deep=True) for msg in initial_system_messages
                    )
                if initial_user_messages:
                    new_messages.extend(
                        msg.model_copy(deep=True) for msg in initial_user_messages
                    )
                elif original_user_messages_text:
                    new_messages.append(
                        ChatMessageUser(content=original_user_messages_text)
                    )
                new_messages.append(ChatMessageUser(content=bridge_prompt))
                state.messages = new_messages

                handoff_count += 1
                handoff_active = False
                if handoff_threshold is not None:
                    next_handoff_token = handoff_threshold
                else:
                    next_handoff_token = None

            num_steps = 0
            start_time = time.time()
            model = get_model()
            setattr(model, "total_retry_time", 0)
            setattr(model, "generate", generate_patched)
            if "o3" in model.api.model_name or "o4" in model.api.model_name or "gpt-5" in model.api.model_name:
                model.api.responses_api = True

            tokenizer_model_name = getattr(getattr(model, "api", None), "model_name", None)
            if not tokenizer_model_name:
                tokenizer_model_name = getattr(model, "name", None)
            if not tokenizer_model_name and hasattr(model, "__class__"):
                tokenizer_model_name = model.__class__.__name__
            tokenizer_model_name_str = str(tokenizer_model_name or "unknown-model")
            conversation_tokenizer = _resolve_conversation_tokenizer(tokenizer_model_name_str)

            # Ensure we have an initial snapshot before the agent starts working.
            _persist_state_snapshot(state, status="initialized")

            # main loop (state.completed checks message_limit and token_limit)
            while not state.completed:
                try:
                    # Proactively prune messages if we're approaching 1000 messages for Claude 3.5 Sonnet on Anthropic API
                    if "sonnet" in model.name.lower() and len(state.messages) > 900:
                        state.messages = prune_messages(state.messages)

                    tokenizer = conversation_tokenizer or _resolve_conversation_tokenizer(
                        tokenizer_model_name_str
                    )
                    current_context_tokens = estimate_conversation_tokens(
                        state.messages, tokenizer
                    )

                    triggered_handoff = False
                    if (
                        handoff_threshold is not None
                        and next_handoff_token is not None
                        and not handoff_active
                        and current_context_tokens >= next_handoff_token
                        and (max_handoffs is None or handoff_count < max_handoffs)
                    ):
                        logger.info(
                            "Context tokens %s reached handoff threshold %s. Requesting summary handoff.",
                            current_context_tokens,
                            next_handoff_token,
                        )
                        state.messages.append(ChatMessageUser(content=summary_prompt))
                        handoff_active = True
                        triggered_handoff = True

                    # Finish depending on step and time limits. If both are set, both must be met.
                    num_steps += 1

                    # NOTE log.warning is necessary here for this to appear in inspect.log, which is necessary so we can know the agents runtime
                    logger.warning(
                        f"total runtime: {round(time.time() - start_time, 2)}, total productive runtime: {round(time.time() - start_time - model.total_retry_time, 2)}, total retry time: {round(model.total_retry_time, 2)}"
                    )

                    over_time_limit = (
                        (time.time() - start_time - model.total_retry_time > real_time_limit)
                        if real_time_limit is not None
                        else False
                    )
                    if real_time_limit is not None and over_time_limit:
                        state.completed = True
                        break

                    # Send progress update every 5 steps
                    if num_steps % 5 == 0 and not triggered_handoff:
                        if real_time_limit is not None:
                            elapsed_time = time.time() - start_time - model.total_retry_time
                            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(real_time_limit)}. Remember, you only have to stop working when the time limit has been reached."
                        else:
                            elapsed_time = time.time() - start_time
                            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed"
                        periodic_msg += "\n\nNote: Don't forget to git commit regularly!"
                        state.messages.append(ChatMessageUser(content=periodic_msg))

                    length_finish_error = False
                    prune_individual = False
                    try:
                        # Calculate timeout in seconds for model.generate if real_time_limit is set
                        generate_timeout = None
                        if real_time_limit is not None:
                            generate_timeout = int(
                                real_time_limit - (time.time() - start_time - model.total_retry_time)
                            )

                        # generate output and append assistant message
                        state.output = await model.generate(
                            self=model,
                            input=state.messages,
                            tools=state.tools,
                            cache=cache,
                            config=GenerateConfig(timeout=generate_timeout, max_retries=20),
                        )
                        state.messages.append(state.output.message)
                        _log_reasoning_usage(state.output)
                    except (LengthFinishReasonError, IndexError) as e:
                        length_finish_error = True
                        if "PRUNE_INDIVIDUAL_MESSAGES" in str(e):
                            prune_individual = True
                    except RetryError as e:
                        # Retry exhausted (e.g., persistent rate limiting). If this may be
                        # driven by oversized context, prune messages and continue.
                        logger.warning(f"retry exhausted: {e}; pruning context and retrying")
                        length_finish_error = True
                        prune_individual = True
                    except JSONDecodeError:
                        state.messages.append(ChatMessageUser(content="The JSON returned was invalid."))
                    except Exception as e:
                        # Check if this is a budget exceeded exception
                        if "BudgetExceededException" in str(type(e)) or "Budget limit" in str(e):
                            logger.error(f"Budget limit exceeded, terminating run: {e}")
                            state.messages.append(ChatMessageUser(content=f"Run terminated: {e}"))
                            state.completed = True
                            break
                        else:
                            # Re-raise other exceptions
                            raise e

                    # Handle context length overflow by pruning messages
                    if length_finish_error or state.output.stop_reason == "model_length":
                        logger.warning("context length overflow")
                        state.messages = prune_messages(
                            state.messages, prune_individual=prune_individual
                        )
                        if handoff_active:
                            # We still want to move forward with a summary even if the model couldn't respond.
                            logger.warning(
                                "Summary handoff failed due to length overflow; using placeholder summary."
                            )
                            perform_handoff(
                                "Summary unavailable: context length was exceeded before the summary could be generated.",
                                context_tokens=current_context_tokens,
                            )
                            continue
                        continue

                    if handoff_active:
                        perform_handoff(
                            state.output.message.text, context_tokens=current_context_tokens
                        )
                        # Skip tool call handling; the next loop iteration restarts fresh.
                        continue

                    # resolve tools calls (if any)
                    if state.output.message.tool_calls:
                        # For each tool call, use timeout equal to the time remaining on this task
                        timeout = None
                        if real_time_limit is not None:
                            timeout = int(
                                real_time_limit - (time.time() - start_time - model.total_retry_time)
                            )

                        # call tool functions
                        try:
                            if timeout is not None:
                                timeout_ctx = getattr(asyncio, "timeout", None)
                                if timeout_ctx is not None:
                                    async with timeout_ctx(timeout):
                                        tool_results = await call_tools(
                                            state.output.message, state.tools, max_output=max_tool_output
                                        )
                                else:
                                    tool_results = await asyncio.wait_for(
                                        call_tools(
                                            state.output.message, state.tools, max_output=max_tool_output
                                        ),
                                        timeout=timeout,
                                    )
                            else:
                                tool_results = await call_tools(
                                    state.output.message, state.tools, max_output=max_tool_output
                                )
                        except asyncio.TimeoutError:
                            state.messages.extend(
                                _tool_timeout_messages(state.output.message.tool_calls or [])
                            )
                            state.messages.append(
                                ChatMessageUser(content="Timeout: The tool call timed out.")
                            )
                            state.completed = True
                            break

                        state.messages.extend(tool_results)

                        # was an answer submitted?
                        answer = submission(tool_results)
                        if answer:
                            # set the output to the answer for scoring
                            state.output.completion = answer

                            # exit if we are at max_attempts
                            attempts += 1
                            if attempts >= max_attempts:
                                state.completed = True
                                break

                            # exit if the submission is successful
                            answer_scores = await score(state)
                            if score_value_fn(answer_scores[0].value) == 1.0:
                                state.completed = True
                                break

                            # otherwise notify the model that it was incorrect and continue
                            else:
                                response_message = (
                                    incorrect_message(state, answer_scores)
                                    if callable(incorrect_message)
                                    else incorrect_message
                                )
                                state.messages.append(ChatMessageUser(content=response_message))

                    # no tool calls, urge the model to continue
                    else:
                        state.messages.append(ChatMessageUser(content=continue_message))
                finally:
                    _persist_state_snapshot(state, status="running")

            return state

        return solve

    # Optional: resume seed solver that rehydrates prior messages from a JSON transcript
    @solver
    def resume_seed() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            resume_path = os.environ.get("RG_RESUME_CONTEXT_FILE", "").strip()
            if not resume_path:
                return state
            try:
                data = json.loads(Path(resume_path).read_text())
                loaded: list[ChatMessage] = []
                for msg in data:
                    role = msg.get("role")
                    if role == "system":
                        loaded.append(ChatMessageSystem(**msg))
                    elif role == "user":
                        loaded.append(ChatMessageUser(**msg))
                    elif role == "assistant":
                        loaded.append(ChatMessageAssistant(**msg))
                    elif role == "tool":
                        loaded.append(ChatMessageTool(**msg))
                if loaded:
                    state.messages = _prune_incomplete_tool_calls(loaded)
            except Exception:
                # If transcript parsing fails, leave state as-is
                pass
            return state

        return solve

    # Helper: persist transcript after each loop iteration if requested
    def _maybe_write_transcript(state: TaskState) -> None:
        transcript_path = os.environ.get("RG_TRANSCRIPT_PATH", "").strip()
        if not transcript_path:
            return
        try:
            path = Path(transcript_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            serializable = [m.model_dump(exclude_none=True) for m in state.messages]  # type: ignore[attr-defined]
            path.write_text(json.dumps(serializable, indent=2))
        except Exception:
            pass

    def _persist_state_snapshot(state: TaskState, status: str | None = None) -> None:
        _maybe_write_transcript(state)
        _maybe_stream_metadata(state, status=status)

    def _maybe_stream_metadata(state: TaskState, status: str | None = None) -> None:
        stream_path = os.environ.get("RG_METADATA_STREAM_PATH", "").strip()
        if not stream_path:
            log_dir = os.environ.get("RG_LOG_DIR", "").strip()
            if log_dir:
                stream_path = str(Path(log_dir) / "metadata_stream.jsonl")
        if not stream_path:
            return
        try:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_started_at": RUN_STARTED_AT_ISO,
                "elapsed_seconds": max(0.0, time.time() - _RUN_STARTED_MONO),
                "status": status or "running",
                "message_count": len(state.messages),
                "token_usage": state.token_usage,
            }
            usage_snapshot: dict[str, dict] = {}
            for model_name, usage in inspect_model_usage().items():
                usage_snapshot[model_name] = usage.model_dump(exclude_none=True)
            payload["model_usage"] = usage_snapshot
            last_usage = getattr(getattr(state, "output", None), "usage", None)
            if last_usage:
                payload["last_call_usage"] = last_usage.model_dump(exclude_none=True)
            path = Path(stream_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _log_reasoning_usage(output: ModelOutput) -> None:
        try:
            usage = getattr(output, "usage", None)
            if not usage or usage.reasoning_tokens is None:
                return
            model_name = getattr(output, "model", "") or "unknown"
            cache_read = usage.input_tokens_cache_read
            cache_str = (
                f", cache_read={cache_read:,}"
                if isinstance(cache_read, int)
                else ""
            )
            print(
                f"[RG] reasoning tokens (model={model_name}): "
                f"in={usage.input_tokens:,}, out={usage.output_tokens:,}, "
                f"reasoning={usage.reasoning_tokens:,}{cache_str}",
                flush=True,
            )
        except Exception:
            pass

    # Wrap basic_agent_loop to add transcript persistence
    original_basic_agent_loop = basic_agent_loop
    @solver
    def basic_agent_loop_with_persist() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # run original loop (support both Solver objects and plain async functions)
            base_solver = original_basic_agent_loop()
            if hasattr(base_solver, "solve"):
                result_state = await base_solver.solve(state, generate)  # type: ignore[attr-defined]
            else:
                result_state = await base_solver(state, generate)  # type: ignore[misc]
            _persist_state_snapshot(
                result_state,
                status="completed" if getattr(result_state, "_completed", False) else "stopped",
            )
            return result_state

        return solve

    # return chain
    if disallow_submit:
        steps: list[Solver] = init + [tools]
        if os.environ.get("RG_RESUME_CONTEXT_FILE"):
            steps.append(resume_seed())
        steps.append(basic_agent_loop_with_persist())
        return chain(steps)
    else:
        steps: list[Solver] = init + [tools, end_task_tool()]
        if os.environ.get("RG_RESUME_CONTEXT_FILE"):
            steps.append(resume_seed())
        steps.append(basic_agent_loop_with_persist())
        return chain(steps)
