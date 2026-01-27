from __future__ import annotations

import asyncio
import copy
import os
import time
from json import JSONDecodeError
from logging import getLogger
from typing import Callable, cast

from inspect_ai._util.format import format_progress_time
from inspect_ai.model import GenerateConfig
from inspect_ai.model._cache import CachePolicy
from inspect_ai.model._call_tools import call_tools
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.model._model import get_model
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


def _tool_timeout_messages(tool_calls: list | None) -> list[ChatMessageTool]:
    """Emit placeholder tool outputs so transcript history stays consistent."""
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


DEFAULT_SYSTEM_MESSAGE = """You are a helpful agent attempting to solve a task.
- You have several functions available to help with solving the task. Each message may perform one function call. You will see the result of the function right after sending the message. If you need to perform multiple actions, you can always send more messages with subsequent function calls.
- Do some reasoning before your actions, describing what function calls you are going to use and how they fit into your plan.
- Use as much of the available time as possible to refine and optimize your solution: Your goal is to achieve the best possible score, not just to submit your first working solution. Keep working until the time limit expires.
- The file read tool is paginated, so you will need to call it multiple times to make sure that you have read all the relevant parts of a paper.
- This is a multi-step task. DO THINGS BIT BY BIT by slowly building up your solution with multiple tool calls, don't try to do everything in one go. Just reason about what is the best next step and execute it. Then take the next step after you receive the output of the previous step.
- You should try and create a clean codebase. When it makes sense you should edit/overwrite files rather than creating new ones.
- When writing code, do not write newlines as the string "\n", instead use the newline character."""

DEFAULT_INCORRECT_MESSAGE = """
Your submission was incorrect. Please proceed and attempt to find the correct answer.
"""
DEFAULT_CONTINUE_MESSAGE = (
    "ERROR; please use the available tools to write and execute the necessary code."
)
DEFAULT_SUBMIT_NAME = "end_task"
DEFAULT_SUBMIT_DESCRIPTION = (
    "Signal that you are completely finished and that you would like to end the task."
)

CONTINUE_USER_MESSAGE = """Now given the previous progress made by the agent, take the next step towards to either propose a new idea to improve performance or refine your previous idea:
- You have a lot of time available, so don't try and do everything in one go. Just reason about what is the best next step and execute it.
- Use the available tools to write and execute the necessary code.
- Remember, you should try prioritize getting results on the required evaluations, and then attempt to improve."""


class BasicAgentDeprecatedArgs(TypedDict, total=False):
    max_messages: int | None


@solver
def basic_agent_iterative(
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
    for arg, value in kwargs.items():
        if arg == "max_messages":
            message_limit = int(cast(int, value))

    if init is None:
        init = system_message(DEFAULT_SYSTEM_MESSAGE)
    init = init if isinstance(init, list) else [init]

    continue_user_message =  CONTINUE_USER_MESSAGE

    if tools is None:
        tools = []
    tools = tools if isinstance(tools, Solver) else use_tools(tools)

    score_value_fn = score_value or value_to_float()

    @tool
    def end_task() -> Tool:
        async def execute(end_message: str) -> ToolResult:
            return end_message

        return execute

    @solver
    def end_task_tool() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.tools.append(tool_with(end_task(), submit_name, submit_description))
            return state

        return solve

    def submission(tool_results: list[ChatMessageTool]) -> str | None:
        return next(
            (result.text for result in tool_results if result.function == submit_name),
            None,
        )

    @solver
    def basic_agent_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.message_limit = message_limit or state.message_limit
            state.token_limit = token_limit or state.token_limit

            attempts = 0
            num_steps = 0
            start_time = time.time()
            model = get_model()
            setattr(model, "total_retry_time", 0)
            setattr(model, "generate", generate_patched)
            if hasattr(model, "api") and (
                "o3" in getattr(model.api, "model_name", "") or "o4" in getattr(model.api, "model_name", "") or "gpt-5" in getattr(model.api, "model_name", "")
            ):
                setattr(model.api, "responses_api", True)

            while not state.completed:
                if "sonnet" in getattr(model, "name", "").lower() and len(state.messages) > 900:
                    state.messages = prune_messages(state.messages)

                num_steps += 1

                length_finish_error = False
                prune_individual = False
                try:
                    conversation = copy.deepcopy(state.messages) + [
                        ChatMessageUser(content=continue_user_message)
                    ]

                    generate_timeout = None
                    if real_time_limit is not None:
                        generate_timeout = int(
                            real_time_limit - (time.time() - start_time - model.total_retry_time)
                        )

                    state.output = await model.generate(
                        self=model,
                        input=conversation,
                        tools=state.tools,
                        cache=cache,
                        config=GenerateConfig(timeout=generate_timeout),
                    )
                    state.messages.append(state.output.message)
                except (LengthFinishReasonError, IndexError) as e:
                    length_finish_error = True
                    prune_individual = False
                    if "PRUNE_INDIVIDUAL_MESSAGES" in str(e):
                        prune_individual = True
                except RetryError as e:
                    # Retry exhausted (commonly due to rate limit). If the underlying
                    # issue is oversized context, proactively prune and continue.
                    logger.warning(f"retry exhausted: {e}; pruning context and retrying")
                    length_finish_error = True
                    prune_individual = True
                except JSONDecodeError:
                    state.messages.append(ChatMessageUser(content="The JSON returned was invalid."))
                    continue

                if length_finish_error or getattr(state.output, "stop_reason", None) == "model_length":
                    logger.warning("context length overflow")
                    state.messages = prune_messages(
                        state.messages, prune_individual=prune_individual
                    )
                    continue

                if state.output.message.tool_calls:
                    timeout = None
                    if real_time_limit is not None:
                        timeout = int(
                            real_time_limit - (time.time() - start_time - model.total_retry_time)
                        )

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
                                        state.output.message,
                                        state.tools,
                                        max_output=max_tool_output,
                                    ),
                                    timeout=timeout,
                                )
                        else:
                            tool_results = await call_tools(
                                state.output.message, state.tools, max_output=max_tool_output
                            )
                    except Exception:
                        state.messages.extend(
                            _tool_timeout_messages(state.output.message.tool_calls or [])
                        )
                        state.messages.append(
                            ChatMessageUser(content="Timeout: The tool call timed out.")
                        )
                        state.completed = True
                        break
                    state.messages.extend(tool_results)
                else:
                    state.messages.append(ChatMessageUser(content=continue_message))

            return state

        return solve

    return chain(
        init
        + [
            tools,
            basic_agent_loop(),
        ]
    )


