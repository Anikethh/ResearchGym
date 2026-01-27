from functools import wraps
from typing import Dict, Optional, List
import tiktoken
from collections import defaultdict
import asyncio
from datetime import datetime
import logging

# Import the enhanced cost tracker
try:
    from .cost_tracker import ai_scientist_cost_tracker
except ImportError:
    # Fallback if cost_tracker is not available
    ai_scientist_cost_tracker = None


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = {
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,  # $1.10 per 1M tokens
                "cached": 0.55 / 1000000,  # $0.55 per 1M tokens
                "completion": 4.4 / 1000000,  # $4.40 per 1M tokens
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens
        
        # Also add to the new enhanced cost tracker if available
        if ai_scientist_cost_tracker is not None:
            try:
                ai_scientist_cost_tracker.add_usage(
                    model=model,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cached_input_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            except Exception as e:
                logging.getLogger("ai-scientist").error(f"Error adding to enhanced cost tracker: {e}")

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        if model not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {model}")
            return 0.0

        prices = self.MODEL_PRICES[model]
        tokens = self.token_counts[model]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )

        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        result = await func(*args, **kwargs)
        model = result.model
        timestamp = result.created

        if hasattr(result, "usage"):
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # Assumes response is in content field
                timestamp,
            )
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract context for tracking more safely
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        
        # If not found in kwargs, try to extract from messages
        if not prompt or not system_message:
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list):
                try:
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role")
                            content = msg.get("content", "")
                            if role == "system" and not system_message:
                                system_message = content
                            elif role == "user" and not prompt:
                                prompt = content
                except Exception:
                    pass  # Ignore extraction errors
        
        result = func(*args, **kwargs)
        
        # Safely extract model and timestamp
        model = getattr(result, 'model', 'unknown')
        timestamp = datetime.fromtimestamp(getattr(result, 'created', 0))
        
        # Token extraction with comprehensive error handling
        try:
            if hasattr(result, "usage") and result.usage:
                usage = result.usage
                
                # Extract basic token counts
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                
                # Extract reasoning tokens safely
                reasoning_tokens = 0
                try:
                    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
                except Exception:
                    pass
                
                # Extract cached tokens safely
                cached_tokens = 0
                try:
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
                except Exception:
                    pass
                
                # Add tokens to tracker
                token_tracker.add_tokens(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    cached_tokens,
                )
                
                # Extract response content safely
                response_content = ""
                try:
                    if hasattr(result, "choices") and result.choices:
                        choice = result.choices[0]
                        if hasattr(choice, "message") and hasattr(choice.message, "content"):
                            response_content = str(choice.message.content or "")
                except Exception:
                    pass
                
                # Add interaction details
                token_tracker.add_interaction(
                    model,
                    str(system_message or ""),
                    str(prompt or ""),
                    response_content,
                    timestamp,
                )
                
                # Log successful tracking (but don't fail if logging fails)
                try:
                    logging.getLogger("ai-scientist").info(
                        f"Tracked tokens for {model}: {prompt_tokens} prompt, {completion_tokens} completion"
                    )
                except Exception:
                    pass
            
        except Exception as e:
            # Log the error but don't fail the function call
            try:
                logging.getLogger("ai-scientist").warning(f"Token tracking failed: {e}")
            except Exception:
                pass
        
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
