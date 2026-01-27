import logging
import os
import time
import json

from .utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
    coerce_to_responses_input,
    extract_text_from_responses_output,
)
from funcy import notnone, once, select_values
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from openai.types.responses import (
    Response as ResponsesResult,
    ResponseFunctionToolCall,
)
from rich import print

# Import token tracking
try:
    from ai_scientist.utils.cost_tracker import ai_scientist_cost_tracker
    COST_TRACKER_AVAILABLE = True
except ImportError:
    print("[red]Warning: AI-Scientist cost tracker not available - token usage will not be tracked[/red]")
    ai_scientist_cost_tracker = None
    COST_TRACKER_AVAILABLE = False

logger = logging.getLogger("ai-scientist")

_client: OpenAI | None = None

OPENAI_TIMEOUT_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@once
def _setup_openai_client():
    global _client
    api_key = (
        os.getenv("AZUREAI_OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("AZUREAI_OPENAI_BASE_URL")
        or os.getenv("AZURE_OPENAI_BASE_URL")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
    )
    if base_url and not base_url.rstrip("/").endswith("/openai/v1"):
        base_url = base_url.rstrip("/") + "/openai/v1"
    client_kwargs = {"api_key": api_key, "max_retries": 0}
    if base_url:
        client_kwargs["base_url"] = base_url
    _client = OpenAI(**client_kwargs)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    if _client is None:
        raise RuntimeError(
            "OpenAI client not initialised. Provide OPENAI_API_KEY or AZURE*_OPENAI credentials in the environment."
        )
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    response_input = coerce_to_responses_input(opt_messages_to_list(system_message, user_message))
    t0 = time.time()
    response: ResponsesResult = backoff_create(
        _client.responses.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        input=response_input,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    output_text = response.output_text or extract_text_from_responses_output(response.output)
    in_tokens = getattr(response.usage, "input_tokens", 0)
    out_tokens = getattr(response.usage, "output_tokens", 0)

    if COST_TRACKER_AVAILABLE and ai_scientist_cost_tracker is not None:
        try:
            cached_tokens = getattr(getattr(response.usage, "input_token_details", None), "cached_tokens", 0)
            reasoning_tokens = getattr(getattr(response.usage, "output_token_details", None), "reasoning_tokens", 0)
            model_name = filtered_kwargs.get("model", response.model)
            ai_scientist_cost_tracker.add_usage(
                model=model_name,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cached_input_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                execution_time=req_time,
                system_message=system_message,
                prompt=user_message,
                response=output_text,
            )
            total_cost = ai_scientist_cost_tracker.get_total_cost()
            print(f"[green]  Total cost so far: ${total_cost:.6f}[/green]")
        except Exception as track_err:
            print(f"[red]✗ Error tracking usage in backend: {track_err}[/red]")
            logger.error("Error tracking usage in backend: %s", track_err, exc_info=True)
    else:
        print(f"[yellow]⚠ AI-Scientist cost tracker not available - token usage not tracked[/yellow]")
        logger.warning("AI-Scientist cost tracker not available - token usage not tracked")

    parsed_output: OutputType
    if func_spec is not None:
        function_payload: OutputType = output_text
        for output in response.output or []:
            if isinstance(output, ResponseFunctionToolCall):
                # prioritize the matching function name when available
                if output.name == func_spec.name or function_payload == output_text:
                    function_payload = output.arguments
                    break
        parsed_output = _coerce_function_output(function_payload, func_spec.name)
    else:
        parsed_output = output_text

    info = {
        "model": response.model,
        "created": response.created_at,
        "id": response.id,
    }

    return parsed_output, req_time, in_tokens, out_tokens, info


def _strip_code_fence(raw: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    stripped = raw.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        # Drop the opening fence (optionally with language tag) and closing fence
        fence_lines = stripped.splitlines()
        if len(fence_lines) >= 2:
            fence_lines = fence_lines[1:-1]
            return "\n".join(fence_lines).strip()
    return stripped


def _coerce_function_output(output: OutputType, func_name: str) -> dict:
    """Ensure function-call responses are returned as dictionaries."""
    if isinstance(output, dict):
        return output
    if output is None:
        raise ValueError(f"Function '{func_name}' returned no content.")
    if not isinstance(output, str):
        raise TypeError(f"Unexpected response type for function '{func_name}': {type(output)}")

    cleaned = _strip_code_fence(output)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to decode function '%s' output as JSON. Raw content: %s",
            func_name,
            cleaned,
        )
        raise ValueError(
            f"Function '{func_name}' response was not valid JSON. Ensure the model returns a JSON object matching the schema."
        ) from exc
