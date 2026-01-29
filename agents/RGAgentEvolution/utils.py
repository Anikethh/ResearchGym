from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from typing import List, cast

import tiktoken
from inspect_ai._util.constants import HTTP
from inspect_ai._util.hooks import send_telemetry
from inspect_ai._util.trace import trace_action
from inspect_ai._util.working import report_sample_waiting_time, sample_working_time
from inspect_ai.model._cache import CacheEntry, CachePolicy, cache_fetch, cache_store
from inspect_ai.model._call_tools import tool_call_view, tools_info
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig, active_generate_config
from inspect_ai.model._model import (
    active_model,
    collapse_consecutive_assistant_messages,
    collapse_consecutive_user_messages,
    handle_sample_message_limit,
    record_model_usage,
    resolve_reasoning_history,
    resolve_tool_model_input,
    tool_result_images_as_user_message,
)
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput
from inspect_ai.tool import Tool, ToolChoice, ToolFunction, ToolInfo
from inspect_ai.tool._tool_def import tool_defs
from openai import LengthFinishReasonError, RateLimitError
from openai.types.chat import ChatCompletion
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, stop_after_delay, stop_never, wait_exponential_jitter
from tenacity.stop import StopBaseT
import logging


logger = logging.getLogger(__name__)

# Global cost tracking for the session
_session_total_cost = 0.0

# Model pricing (per 1M tokens)
_MODEL_PRICES = {
    "gpt-5": {
        "input": 1.25,  # $1.25 per 1M tokens
        "cached_input": 0.125,  # $0.125 per 1M tokens
        "output": 10.00,  # $10.00 per 1M tokens
    },
    "google/gemini-2.5-pro": {
        "input_low": 1.25,    # $1.25 per 1M tokens (≤200k)
        "input_high": 2.50,   # $2.50 per 1M tokens (>200k)
        "output_low": 10.00,  # $10.00 per 1M tokens (≤200k)
        "output_high": 15.00, # $15.00 per 1M tokens (>200k)
        "cached_low": 0.31,   # $0.31 per 1M tokens (≤200k)
        "cached_high": 0.625, # $0.625 per 1M tokens (>200k)
        "threshold": 200000,  # 200k token threshold
    },
    "google/gemini-2.5-flash-lite": {
        "input": 0.10,   # $0.10 per 1M tokens
        "output": 0.40,  # $0.40 per 1M tokens
        "cached_input": 0.025,  # $0.025 per 1M tokens
    },
}


def _calculate_request_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """Calculate cost for a single request."""
    
    # Normalize model name for pricing lookup
    normalized_model = model
    if "gpt-5" in model:
        normalized_model = "gpt-5"
    elif "gemini-2.5-pro" in model:
        normalized_model = "google/gemini-2.5-pro"
    elif "gemini-2.5-flash-lite" in model:
        normalized_model = "google/gemini-2.5-flash-lite"
    
    if normalized_model not in _MODEL_PRICES:
        logger.warning(f"No pricing information for model: {model} (normalized: {normalized_model})")
        return 0.0
        
    prices = _MODEL_PRICES[normalized_model]
    cost = 0.0
    
    # Handle tiered pricing for Gemini 2.5 Pro
    if normalized_model == "google/gemini-2.5-pro":
        threshold = prices["threshold"]
        
        # Calculate actual new input tokens (subtract cached from total)
        new_input_tokens = input_tokens - cached_input_tokens
        
        # Input tokens cost (tiered, only for NEW tokens)
        if new_input_tokens > 0:
            if new_input_tokens <= threshold:
                cost += new_input_tokens * prices["input_low"] / 1_000_000
            else:
                cost += threshold * prices["input_low"] / 1_000_000
                cost += (new_input_tokens - threshold) * prices["input_high"] / 1_000_000
        
        # Output tokens cost (tiered)
        if output_tokens <= threshold:
            cost += output_tokens * prices["output_low"] / 1_000_000
        else:
            cost += threshold * prices["output_low"] / 1_000_000
            cost += (output_tokens - threshold) * prices["output_high"] / 1_000_000
        
        # Cached tokens cost (tiered, at discounted rates)
        if cached_input_tokens > 0:
            if cached_input_tokens <= threshold:
                cost += cached_input_tokens * prices["cached_low"] / 1_000_000
            else:
                cost += threshold * prices["cached_low"] / 1_000_000
                cost += (cached_input_tokens - threshold) * prices["cached_high"] / 1_000_000
    
    # Handle flat pricing for other models
    else:
        # Calculate actual new input tokens (subtract cached from total)
        new_input_tokens = input_tokens - cached_input_tokens
        
        # Regular input tokens (only the new ones)
        if "input" in prices and new_input_tokens > 0:
            cost += new_input_tokens * prices["input"] / 1_000_000
        
        # Cached input tokens (at discounted rate)
        if cached_input_tokens > 0 and "cached_input" in prices:
            cost += cached_input_tokens * prices["cached_input"] / 1_000_000
        
        # Output tokens (includes reasoning for O1 models)
        if "output" in prices:
            cost += output_tokens * prices["output"] / 1_000_000
            
    return cost


def _track_session_cost(cost: float) -> None:
    """Add cost to session total."""
    global _session_total_cost
    _session_total_cost += cost


def _get_session_total_cost() -> float:
    """Get total cost for this session."""
    return _session_total_cost


def handle_message_len(
    msg: ChatMessage,
    tokenizer,
    max_tokens: int,
) -> ChatMessage:
    def truncate_string(input_string: str, input_tokens: list, max_tokens: int) -> str:
        n_tokens = len(input_tokens)
        if n_tokens > max_tokens:
            keep_tokens = max_tokens // 2
            first_half = tokenizer.decode(input_tokens[:keep_tokens])
            second_half = tokenizer.decode(input_tokens[-keep_tokens:])
            return first_half + "\n...[content truncated due to length]...\n" + second_half
        return input_string

    if isinstance(msg.content, str):
        item_tokens = tokenizer.encode(msg.content, disallowed_special=())
        msg.content = truncate_string(msg.content, item_tokens, max_tokens)
    elif isinstance(msg.content, list):
        token_lists: list[list[int]] = []
        token_counts: list[int] = []
        for item in msg.content:
            if item.type == "text":
                item_tokens = tokenizer.encode(item.text, disallowed_special=())
                token_lists.append(item_tokens)
                token_counts.append(len(item_tokens))
            elif item.type == "reasoning":
                item_tokens = tokenizer.encode(item.reasoning, disallowed_special=())
                token_lists.append(item_tokens)
                token_counts.append(len(item_tokens))
            else:
                token_lists.append([])
                token_counts.append(0)

        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return msg

        tokens_per_item = [
            max(1, int((count / total_tokens) * max_tokens)) if count > 0 else 0
            for count in token_counts
        ]

        new_content = []
        for item, item_tokens, max_tokens_for_item in zip(
            msg.content, token_lists, tokens_per_item
        ):
            if item.type == "text":
                item.text = truncate_string(item.text, item_tokens, max_tokens_for_item)
            elif item.type == "reasoning":
                item.reasoning = truncate_string(item.reasoning, item_tokens, max_tokens_for_item)
            new_content.append(item)

        msg.content = new_content

    return msg


def get_gpu_generation() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    generation = result.stdout.strip().split("\n")
    if not generation:
        return None
    return ", ".join([info.strip() for info in generation])


def append_system_message(messages: list[ChatMessage], message: ChatMessageSystem) -> None:
    lastIndex = -1
    for i in list(reversed(range(0, len(messages)))):
        if isinstance(messages[i], ChatMessageSystem):
            lastIndex = i
            break
    messages.insert(lastIndex + 1, message)


def _env_int(name: str, default: int) -> int:
    try:
        val = os.environ.get(name, None)
        return int(val) if val is not None and str(val).strip() != "" else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        val = os.environ.get(name, None)
        return float(val) if val is not None and str(val).strip() != "" else default
    except Exception:
        return default


def prune_messages(
    messages: List[ChatMessage], prune_individual: bool = False
) -> List[ChatMessage]:
    system_msgs: List[ChatMessage] = [m for m in messages if m.role == "system"]
    conversation = [m for m in messages if m.role != "system"]
    task_msg = next((m for m in conversation if m.role == "user"), None)
    # Allow tuning of pruning ratio via env var (default keep last 70%).
    prune_start_ratio = _env_float("RG_PRUNE_START_RATIO", 0.3)
    prune_start_ratio = min(max(prune_start_ratio, 0.0), 0.9)
    start_idx = max(1, int(len(conversation) * prune_start_ratio))
    preserved: List[ChatMessage] = [task_msg] if task_msg else []
    preserved.extend(conversation[start_idx:])
    conversation = preserved
    valid_messages = []
    active_tool_ids = set()
    for msg in conversation:
        if "prompt is too long" in msg.content:
            continue
        if msg.role == "assistant":
            active_tool_ids = {tc.id for tc in (msg.tool_calls or [])}
            valid_messages.append(msg)
        elif msg.role == "tool" and getattr(msg, "tool_call_id", None) in active_tool_ids:
            valid_messages.append(msg)
        elif msg.role == "user":
            active_tool_ids = set()
            valid_messages.append(msg)
    if prune_individual:
        # Allow tuning per-message cap via env var; default 100k tokens.
        MAX_TOKENS_PER_MESSAGE = _env_int("RG_MAX_TOKENS_PER_MESSAGE", 100000)
        tokenizer = tiktoken.get_encoding("o200k_base")
        valid_messages = [
            handle_message_len(msg, tokenizer, MAX_TOKENS_PER_MESSAGE) for msg in valid_messages
        ]
    return cast(List[ChatMessage], system_msgs + valid_messages)


def log_rate_limit_retry(context: str, retry_state: RetryCallState) -> None:
    logger.log(
        HTTP,
        f"{context} rate limit retry {retry_state.attempt_number} after waiting for {retry_state.idle_for}",
    )


async def generate_patched(
    self,
    input: str | list[ChatMessage],
    tools: list[Tool] | list[ToolDef] | list[ToolInfo] | list[Tool | ToolDef | ToolInfo] = [],
    tool_choice: ToolChoice | None = None,
    config: GenerateConfig = GenerateConfig(),
    cache: bool | CachePolicy = False,
) -> ModelOutput:
    is_active_model = self == active_model()
    if is_active_model:
        handle_sample_message_limit(input)
    base_config = self.config
    if is_active_model:
        base_config = base_config.merge(active_generate_config())
    config = base_config.merge(config)
    if config.max_tokens is None:
        config.max_tokens = self.api.max_tokens_for_config(config)
        if config.max_tokens is None:
            config.max_tokens = self.api.max_tokens()
    if isinstance(input, str):
        input = [ChatMessageUser(content=input)]
    if config.system_message:
        input = [ChatMessageSystem(content=config.system_message)] + input
    start_time = datetime.now()
    working_start = sample_working_time()
    from inspect_ai.log._samples import track_active_sample_retries

    with track_active_sample_retries():
        output = await _generate(
            self=self,
            input=input,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
            cache=cache,
        )

    from inspect_ai.log._transcript import ModelEvent, transcript

    last_model_event = transcript().find_last_event(ModelEvent)
    if last_model_event:
        last_model_event.timestamp = start_time
        last_model_event.working_start = working_start
        completed = datetime.now()
        last_model_event.completed = completed
        last_model_event.working_time = (
            output.time if output.time is not None else (completed - start_time).total_seconds()
        )
    return output


async def _generate(
    self,
    input: list[ChatMessage],
    tools: list[Tool] | list[ToolDef] | list[ToolInfo] | list[Tool | ToolDef | ToolInfo],
    tool_choice: ToolChoice | None,
    config: GenerateConfig,
    cache: bool | CachePolicy = False,
) -> ModelOutput:
    tool_choice = tool_choice if tool_choice else "auto"
    tdefs = tool_defs([tool for tool in tools if not isinstance(tool, ToolInfo)])
    tools = tools_info(tools)
    if isinstance(tool_choice, ToolFunction):
        tools = [tool for tool in tools if tool.name == tool_choice.name]
    if tool_choice == "none" or len(tools) == 0:
        if not self.api.tools_required():
            tools = []
        tool_choice = "none"
    input = resolve_reasoning_history(input, config, self.api)
    input = resolve_tool_model_input(tdefs, input)
    if not self.api.tool_result_images():
        input = tool_result_images_as_user_message(input)
    if self.api.collapse_user_messages():
        input = collapse_consecutive_user_messages(input)
    if self.api.collapse_assistant_messages():
        input = collapse_consecutive_assistant_messages(input)
    if config.max_retries is not None and config.timeout is not None:
        stop: StopBaseT = stop_after_attempt(config.max_retries) | stop_after_delay(config.timeout)
    elif config.max_retries is not None:
        stop = stop_after_attempt(config.max_retries)
    elif config.timeout is not None:
        stop = stop_after_delay(config.timeout)
    else:
        stop = stop_never

    def before_sleep(retry_state: RetryCallState) -> None:
        wait_time = retry_state.next_action.sleep
        if hasattr(self, "total_retry_time"):
            self.total_retry_time += wait_time
        logger.log(HTTP, f"{self.api.model_name} rate limit retry {retry_state.attempt_number} after waiting for {retry_state.idle_for}")

    @retry(
        wait=wait_exponential_jitter(initial=6, max=(4 * 60), jitter=15),
        retry=retry_if_exception(self.should_retry),
        stop=stop,
        before_sleep=before_sleep,
    )
    async def generate() -> ModelOutput:
        cache_entry: CacheEntry | None
        if cache:
            if isinstance(cache, CachePolicy):
                policy = cache
            else:
                policy = CachePolicy()
            cache_entry = CacheEntry(
                base_url=self.api.base_url,
                config=json.loads(json.dumps(config.model_dump())),
                input=input,
                model=str(self),
                policy=policy,
                tool_choice=tool_choice,
                tools=tools,  # type: ignore
            )
            existing = cache_fetch(cache_entry)
            if isinstance(existing, ModelOutput):
                self._record_model_interaction(
                    input=input,
                    tools=tools,
                    tool_choice=tool_choice,
                    config=config,
                    cache="read",
                    output=existing,
                    call=None,
                )
                return existing
        else:
            cache_entry = None

        self.verify_model_apis()
        complete = self._record_model_interaction(
            input=input,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
            cache="write" if cache else None,
        )

        with trace_action(logger, "Model", f"generate ({str(self)})"):
            time_start = time.monotonic()
            try:
                if config.timeout is not None:
                    import asyncio

                    timeout_ctx = getattr(asyncio, "timeout", None)
                    if timeout_ctx is not None:
                        async with timeout_ctx(config.timeout):
                            result = await self.api.generate(
                                input=input,
                                tools=tools,
                                tool_choice=tool_choice,
                                config=config,
                            )
                    else:
                        result = await asyncio.wait_for(
                            self.api.generate(
                                input=input,
                                tools=tools,
                                tool_choice=tool_choice,
                                config=config,
                            ),
                            timeout=config.timeout,
                        )
                else:
                    result = await self.api.generate(
                        input=input,
                        tools=tools,
                        tool_choice=tool_choice,
                        config=config,
                    )
            except Exception as e:
                from inspect_ai.model._providers.openrouter import OpenRouterError

                # Map known provider errors indicating context/window overflow to a
                # LengthFinishReasonError so upstream pruning can engage.
                if isinstance(e, OpenRouterError) and (
                    "exceed context limit" in str(e)
                    or "context length" in str(e)
                    or "too long" in str(e)
                ):
                    error_completion = ChatCompletion(
                        choices=[], id="", created=0, model="", object="chat.completion"
                    )
                    error = LengthFinishReasonError(completion=error_completion)
                    if "too long" in str(e):
                        error.args = ("PRUNE_INDIVIDUAL_MESSAGES: Message is too long",)
                    raise error

                # Some OpenAI/Azure endpoints surface oversized prompts as RateLimitError
                # rather than BadRequestError/context_length_exceeded. Detect these and
                # convert to LengthFinishReasonError to trigger pruning.
                if isinstance(e, RateLimitError):
                    msg = str(e)
                    if (
                        "context length" in msg
                        or "maximum context" in msg
                        or "prompt too long" in msg
                        or "too many tokens" in msg
                        or "token limit" in msg
                    ):
                        error_completion = ChatCompletion(
                            choices=[], id="", created=0, model="", object="chat.completion"
                        )
                        error = LengthFinishReasonError(completion=error_completion)
                        # Encourage per-message truncation when the prompt itself is too long
                        error.args = (
                            "PRUNE_INDIVIDUAL_MESSAGES: Message is too long",
                        )
                        raise error

                # Fall through to original exception
                raise
            finally:
                time_elapsed = time.monotonic() - time_start

        if isinstance(result, tuple):
            output, call = result
        else:
            output = result
            call = None

        if isinstance(output, Exception):
            complete(output, call)
            error = repr(output)
            request = json.dumps(call.request, indent=2) if call is not None else ""
            error_message = f"{error}\n\nRequest:\n{request}"
            raise RuntimeError(error_message)

        if call and call.time is not None:
            output.time = call.time
        else:
            output.time = time_elapsed

        for choice in output.choices:
            for tool_call in choice.message.tool_calls or []:
                tool_call.view = tool_call_view(tool_call, tdefs)

        complete(output, call)

        if output.usage:
            record_model_usage(f"{self}", output.usage)
            
            # Add real-time cost tracking and limit checking
            try:
                # Calculate cost for this request
                cost = _calculate_request_cost(
                    model=str(self),
                    input_tokens=output.usage.input_tokens,
                    output_tokens=output.usage.output_tokens,
                    cached_input_tokens=output.usage.input_tokens_cache_read or 0,
                    reasoning_tokens=output.usage.reasoning_tokens or 0,
                )
                
                # Log the cost
                print(f"Request cost: ${cost:.6f} for {str(self)} ({output.usage.input_tokens} in, {output.usage.output_tokens} out, {output.usage.input_tokens_cache_read or 0} cached)")
                logger.info(f"Request cost: ${cost:.6f} for {str(self)} ({output.usage.input_tokens} in, {output.usage.output_tokens} out)")
                
                # Track the cost for session total first
                _track_session_cost(cost)
                session_total = _get_session_total_cost()
                
                # Check budget limit if set (after tracking cost)
                budget_limit = float(os.environ.get("RG_BUDGET_LIMIT", "0"))
                if budget_limit > 0:
                    if session_total > budget_limit:
                        logger.error(f"Budget limit exceeded! Total: ${session_total:.6f}, Limit: ${budget_limit:.6f}")
                        print(f"Budget limit exceeded! Total: ${session_total:.6f}, Limit: ${budget_limit:.6f}")
                        # Create a custom exception type that can be caught and handled appropriately
                        from inspect_ai.solver._basic_agent import BudgetExceededException
                        raise BudgetExceededException(f"Budget limit of ${budget_limit:.6f} exceeded (current: ${session_total:.6f})")
                
                # Log session total
                print(
                    f"Session total cost: ${session_total:.6f} (limit: ${budget_limit:.2f})"
                    if budget_limit > 0
                    else f"Session total cost: ${session_total:.6f}"
                )
                logger.info(f"Session total cost: ${session_total:.6f}")
                
            except Exception as e:
                # Only catch budget exceptions specially if they're the budget exception type
                if "BudgetExceededException" in str(type(e)):
                    # Re-raise budget exceptions so they can be handled at the agent level
                    raise e
                else:
                    logger.warning(f"Cost tracking failed: {e}")
            
            await send_telemetry(
                "model_usage",
                json.dumps(dict(model=str(self), usage=output.usage.model_dump())),
            )
        if cache and cache_entry:
            cache_store(entry=cache_entry, output=output)
        return output

    time_start = time.monotonic()
    model_output = await generate()
    total_time = time.monotonic() - time_start
    if model_output.time:
        report_sample_waiting_time(total_time - model_output.time)
    return model_output
