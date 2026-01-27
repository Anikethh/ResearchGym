from dataclasses import dataclass

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


import backoff
import logging
from typing import Callable, Iterable, List, Dict, Any

logger = logging.getLogger("ai-scientist")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def coerce_to_responses_input(messages: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    """Convert chat-style messages into Responses API input format."""
    responses_input: List[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            normalized_content = []
            for item in content:
                if isinstance(item, dict) and "type" in item:
                    normalized_content.append(item)
                elif isinstance(item, str):
                    normalized_content.append({"type": "input_text", "text": item})
            if not normalized_content:
                normalized_content = [{"type": "input_text", "text": str(content)}]
        elif isinstance(content, dict) and "type" in content:
            normalized_content = [content]
        else:
            normalized_content = [{"type": "input_text", "text": str(content)}]
        responses_input.append({"role": role, "content": normalized_content})
    if not responses_input:
        responses_input.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})
    return responses_input


def extract_text_from_responses_output(response_output: List[Any]) -> str:
    """Flatten textual content from the Responses API output list."""
    parts: List[str] = []
    for item in response_output or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for content_part in item.get("content", []):
                text = content_part.get("text") if isinstance(content_part, dict) else None
                if isinstance(text, dict):
                    value = text.get("value")
                    if isinstance(value, str):
                        parts.append(value)
                elif isinstance(text, str):
                    parts.append(text)
        elif item.get("type") == "output_text":
            text_value = item.get("text")
            if isinstance(text_value, dict) and isinstance(text_value.get("value"), str):
                parts.append(text_value["value"])
            elif isinstance(text_value, str):
                parts.append(text_value)
    return "\n".join(part.strip() for part in parts if part)


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    """Convert a prompt into markdown format"""
    try:
        logger.debug(f"compile_prompt_to_md input: type={type(prompt)}")
        if isinstance(prompt, (list, dict)):
            logger.debug(f"prompt content: {prompt}")

        if prompt is None:
            return ""

        if isinstance(prompt, str):
            return prompt.strip() + "\n"

        if isinstance(prompt, list):
            # Handle empty list case
            if not prompt:
                return ""
            # Special handling for multi-modal messages
            if all(isinstance(item, dict) and "type" in item for item in prompt):
                # For multi-modal messages, just pass through without modification
                return prompt

            try:
                result = "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])
                return result
            except Exception as e:
                logger.error(f"Error processing list items: {e}")
                logger.error("List contents:")
                for i, item in enumerate(prompt):
                    logger.error(f"  Item {i}: type={type(item)}, value={item}")
                raise

        if isinstance(prompt, dict):
            # Check if this is a single multi-modal message
            if "type" in prompt:
                return prompt

            # Regular dict processing
            try:
                out = []
                header_prefix = "#" * _header_depth
                for k, v in prompt.items():
                    logger.debug(f"Processing dict key: {k}")
                    out.append(f"{header_prefix} {k}\n")
                    out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
                return "\n".join(out)
            except Exception as e:
                logger.error(f"Error processing dict: {e}")
                logger.error(f"Dict contents: {prompt}")
                raise

        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    except Exception as e:
        logger.error("Error in compile_prompt_to_md:")
        logger.error(f"Input type: {type(prompt)}")
        logger.error(f"Input content: {prompt}")
        logger.error(f"Error: {str(e)}")
        raise


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Return tool definition compatible with OpenAI Responses API."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
        }

    @property
    def openai_tool_choice_dict(self):
        """Force the model to call this function when tool_choice is set."""
        return {
            "type": "function",
            "name": self.name,
        }
