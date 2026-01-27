import logging

import os
from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


logger = logging.getLogger("ai-scientist.backend")


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    # Temperature control is unsupported for the current models; ignore any provided value.
    model_kwargs = model_kwargs | {
        "model": model,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens
        # Ensure at least one user message for OpenAI-compatible backends (e.g., Gemini via OpenAI API)
        if system_message is not None and user_message is None:
            user_message = system_message
            system_message = None

    # Compose a generic system preamble, similar to Basic-Agent, to guide research behavior
    generic_preamble = os.getenv(
        "RG_AI_SCI_SYSTEM_MESSAGE",
        (
            "You are an autonomous research agent. Work iteratively, justify your steps, "
            "and implement code to test ideas. Use time and API budget responsibly; stop "
            "when either is exhausted. Prefer reproducible commands and clear logging. "
            "Follow the task documentation faithfully—use the provided datasets, scripts, and metrics, "
            "and do not fabricate new benchmarks."
        ),
    ).strip()

    # Compile prompts to Markdown/text and prepend the generic preamble appropriately
    compiled_system = compile_prompt_to_md(system_message) if system_message else None
    compiled_user = compile_prompt_to_md(user_message) if user_message else None

    if isinstance(compiled_system, str):
        compiled_system = generic_preamble + "\n\n" + compiled_system
    elif compiled_system is None and isinstance(compiled_user, str):
        compiled_user = generic_preamble + "\n\n" + compiled_user
    elif compiled_system is None and compiled_user is None:
        compiled_system = generic_preamble

    query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compiled_system,
        user_message=compiled_user,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
