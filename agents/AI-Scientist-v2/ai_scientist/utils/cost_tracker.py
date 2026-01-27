"""
Enhanced cost tracking module for AI-Scientist with ResearchGym integration.
Supports cached tokens, detailed cost calculations, and time tracking.
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Any, List

try:  # POSIX-only; gracefully degrade on other platforms
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows compatibility
    fcntl = None

import os

logger = logging.getLogger(__name__)


class AIScientistCostTracker:
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize cost tracker with model pricing information.
        
        Args:
            log_file: Optional path to save cost logs
        """
        self.token_counts = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            }
        )
        self.cost_history = []
        self.interactions = defaultdict(list)
        self.timing_data = defaultdict(list)
        self.log_file = log_file
        self.instance_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._entry_sequence = 0
        
        # Model pricing (per 1M tokens) - aligned with BasicAgent + AI-Scientist models
        self.MODEL_PRICES = {
            # GPT-4o models
            "gpt-4o": {
                "input": 2.50,  # $2.50 per 1M tokens
                "cached_input": 1.25,  # $1.25 per 1M tokens  
                "output": 10.00,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-11-20": {
                "input": 2.50,
                "cached_input": 1.25,
                "output": 10.00,
            },
            "gpt-4o-2024-08-06": {
                "input": 2.50,
                "cached_input": 1.25,
                "output": 10.00,
            },
            "gpt-4o-2024-05-13": {  # this version does not support cached tokens
                "input": 5.00,
                "output": 15.00,
            },
            "gpt-4o-mini": {
                "input": 0.15,
                "cached_input": 0.075,
                "output": 0.60,
            },
            "gpt-4o-mini-2024-07-18": {
                "input": 0.15,
                "cached_input": 0.075,
                "output": 0.60,
            },
            
            # O1 models 
            "o1": {
                "input": 15.00,
                "cached_input": 7.50,
                "output": 60.00,
            },
            "o1-2024-12-17": {
                "input": 15.00,
                "cached_input": 7.50,
                "output": 60.00,
            },
            "o1-preview-2024-09-12": {
                "input": 15.00,
                "cached_input": 7.50,
                "output": 60.00,
            },
            "o1-mini": {
                "input": 3.00,
                "cached_input": 1.50,
                "output": 12.00,
            },
            "o1-mini-2024-09-12": {
                "input": 3.00,
                "cached_input": 1.50,
                "output": 12.00,
            },
            
            # O3 models
            "o3-mini": {
                "input": 1.10,
                "cached_input": 0.55,
                "output": 4.40,
            },
            "o3-mini-2025-01-31": {
                "input": 1.10,
                "cached_input": 0.55,
                "output": 4.40,
            },
            
            # GPT-5 models (for compatibility with BasicAgent)
            "gpt-5": {
                "input": 1.25,
                "cached_input": 0.125,
                "output": 10.00,
            },
            
            # Gemini models (tiered pricing)
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
                "input": 0.10,
                "output": 0.40,
                "cached_input": 0.025,
            },
            
            # Claude models (approximate pricing)
            "claude-3-5-sonnet-20240620": {
                "input": 3.00,
                "output": 15.00,
            },
            "claude-3-5-sonnet-20241022": {
                "input": 3.00,
                "output": 15.00,
            },
        }

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        total_tokens: Optional[int] = None,
        execution_time: Optional[float] = None,
        system_message: Optional[str] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
    ) -> float:
        """
        Add token usage and calculate cost.
        
        Args:
            model: Model name
            input_tokens: Regular input tokens
            output_tokens: Output tokens
            cached_input_tokens: Cached input tokens
            reasoning_tokens: Reasoning tokens (for O1 models)
            total_tokens: Total tokens (calculated if not provided)
            execution_time: Time taken for the request in seconds
            system_message: System message for interaction tracking
            prompt: User prompt for interaction tracking  
            response: Model response for interaction tracking
            
        Returns:
            Cost for this usage in USD
        """
        # Normalize model name for pricing lookup
        normalized_model = self._normalize_model_name(model)
        
        # Update counts
        self.token_counts[normalized_model]["input_tokens"] += input_tokens
        self.token_counts[normalized_model]["output_tokens"] += output_tokens
        self.token_counts[normalized_model]["cached_input_tokens"] += cached_input_tokens
        self.token_counts[normalized_model]["reasoning_tokens"] += reasoning_tokens
        
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens + cached_input_tokens + reasoning_tokens
        self.token_counts[normalized_model]["total_tokens"] += total_tokens
        
        # Calculate cost
        cost = self._calculate_cost(normalized_model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens)
        
        # Check cost limit if set (similar to BasicAgent)
        import os
        cost_limit = float(os.environ.get("AI_SCIENTIST_COST_LIMIT", "0"))
        if cost_limit > 0:
            current_total = self.get_total_cost()
            new_total = current_total + cost
            if new_total > cost_limit:
                logger.error(f"AI-Scientist cost limit exceeded! Total: ${new_total:.6f}, Limit: ${cost_limit:.6f}")
                raise RuntimeError(f"AI-Scientist cost limit of ${cost_limit:.6f} exceeded (current: ${new_total:.6f})")
        
        # Log this usage
        usage_entry = {
            "id": f"{self.instance_id}-{self._entry_sequence}",
            "tracker_id": self.instance_id,
            "sequence": self._entry_sequence,
            "timestamp": datetime.now().isoformat(),
            "model": normalized_model,
            "original_model": model,  # Keep original model name too
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "execution_time_seconds": execution_time,
            "system_message": system_message,
            "prompt": prompt,
            "response": response,
        }
        self._entry_sequence += 1
        self.cost_history.append(usage_entry)
        
        # Track interaction details if provided
        if system_message or prompt or response:
            self.interactions[normalized_model].append({
                "timestamp": datetime.now().isoformat(),
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "cost_usd": cost,
                "execution_time_seconds": execution_time,
            })
        
        # Track timing data
        if execution_time is not None:
            self.timing_data[normalized_model].append({
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "tokens": total_tokens,
            })
        
        # Save to file if configured
        if self.log_file:
            self._save_to_file([usage_entry])
            
        logger.info(f"AI-Scientist cost tracking - Model: {normalized_model}, Cost: ${cost:.6f}, "
                   f"Input: {input_tokens}, Cached: {cached_input_tokens}, Output: {output_tokens}"
                   f"{f', Time: {execution_time:.2f}s' if execution_time else ''}")
        
        return cost

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for pricing lookup."""
        # Keep original name if it's already in our pricing table
        if model in self.MODEL_PRICES:
            return model
            
        # Try common normalization patterns
        if "gpt-4o-mini" in model and model not in self.MODEL_PRICES:
            return "gpt-4o-mini"
        elif "gpt-4o" in model and model not in self.MODEL_PRICES:
            return "gpt-4o"
        elif "gpt-5" in model:
            return "gpt-5"
        elif "o1-mini" in model and model not in self.MODEL_PRICES:
            return "o1-mini"
        elif "o1" in model and model not in self.MODEL_PRICES:
            return "o1"
        elif "o3-mini" in model and model not in self.MODEL_PRICES:
            return "o3-mini"
        elif "gemini-2.5-pro" in model:
            return "google/gemini-2.5-pro"
        elif "gemini-2.5-flash-lite" in model:
            return "google/gemini-2.5-flash-lite"
        elif "claude-3-5-sonnet" in model:
            return "claude-3-5-sonnet-20241022"  # Use latest version
            
        return model

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> float:
        """Calculate cost for given usage."""
        if model not in self.MODEL_PRICES:
            logger.warning(f"No pricing information for model: {model}")
            return 0.0
            
        prices = self.MODEL_PRICES[model]
        cost = 0.0
        
        # Handle tiered pricing for Gemini 2.5 Pro
        if model == "google/gemini-2.5-pro":
            threshold = prices["threshold"]
            
            # Input tokens cost (tiered)
            if input_tokens <= threshold:
                cost += input_tokens * prices["input_low"] / 1_000_000
            else:
                cost += threshold * prices["input_low"] / 1_000_000
                cost += (input_tokens - threshold) * prices["input_high"] / 1_000_000
            
            # Output tokens cost (tiered)
            if output_tokens <= threshold:
                cost += output_tokens * prices["output_low"] / 1_000_000
            else:
                cost += threshold * prices["output_low"] / 1_000_000
                cost += (output_tokens - threshold) * prices["output_high"] / 1_000_000
            
            # Cached tokens cost (tiered)
            if cached_input_tokens <= threshold:
                cost += cached_input_tokens * prices["cached_low"] / 1_000_000
            else:
                cost += threshold * prices["cached_low"] / 1_000_000
                cost += (cached_input_tokens - threshold) * prices["cached_high"] / 1_000_000
        
        # Handle flat pricing for other models
        else:
            # Regular input tokens
            if "input" in prices:
                cost += input_tokens * prices["input"] / 1_000_000
            
            # Cached input tokens
            if cached_input_tokens > 0 and "cached_input" in prices:
                cost += cached_input_tokens * prices["cached_input"] / 1_000_000
            
            # Output tokens (includes reasoning for O1 models)
            if "output" in prices:
                cost += output_tokens * prices["output"] / 1_000_000
                
        return cost

    def get_total_cost(self, model: Optional[str] = None) -> float:
        """Get total cost for all models or specific model."""
        summary = self.get_summary()
        if model:
            model_summary = summary.get(model, {})
            return float(model_summary.get("cost_usd", 0.0))
        return float(summary.get("total_cost_usd", 0.0))

    def get_summary(self) -> Dict:
        """Get comprehensive summary of usage and costs."""
        log_data = self._load_log_data()
        if log_data and isinstance(log_data, dict):
            summary = log_data.get("summary")
            if isinstance(summary, dict):
                return summary

        # Fallback: compute from local history (single process)
        return self._compute_summary_from_history(self.cost_history)

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        log_data = self._load_log_data()
        if log_data and isinstance(log_data, dict) and "history" in log_data:
            interactions = defaultdict(list)
            for entry in log_data.get("history", []):
                system_message = entry.get("system_message")
                prompt = entry.get("prompt")
                response = entry.get("response")
                if system_message or prompt or response:
                    model_name = entry.get("model", "unknown")
                    interactions[model_name].append(
                        {
                            "timestamp": entry.get("timestamp"),
                            "system_message": system_message,
                            "prompt": prompt,
                            "response": response,
                            "cost_usd": entry.get("cost_usd"),
                            "execution_time_seconds": entry.get("execution_time_seconds"),
                        }
                    )

            if model:
                return {model: interactions.get(model, [])}
            return dict(interactions)

        # Fallback to in-memory interactions (single process usage)
        if model:
            return {model: self.interactions.get(model, [])}
        return dict(self.interactions)

    def _load_log_data(self) -> Optional[Dict[str, Any]]:
        """Load persisted log data if available."""
        if not self.log_file or not self.log_file.exists():
            return None

        try:
            content = self.log_file.read_text()
            if not content.strip():  # Empty file
                return None
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.error(f"Failed to load AI-Scientist cost log: {e}")
        return None

    @staticmethod
    def _initialize_model_summary() -> Dict[str, Any]:
        return {
            "tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            },
            "cost_usd": 0.0,
            "timing": {
                "total_time_seconds": 0.0,
                "average_time_seconds": 0.0,
                "num_requests": 0,
            },
            "interactions_count": 0,
        }

    def _compute_summary_from_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not history:
            return {
                "total_cost_usd": 0.0,
                "total_time_seconds": 0.0,
                "total_entries": 0,
                "total_interactions": 0,
            }

        per_model: Dict[str, Dict[str, Any]] = {}
        total_cost = 0.0
        total_time = 0.0
        total_entries = len(history)
        total_interactions = 0

        for entry in history:
            model_name = entry.get("model") or "unknown"
            if model_name not in per_model:
                per_model[model_name] = self._initialize_model_summary()
            model_summary = per_model[model_name]

            model_summary_tokens = model_summary["tokens"]
            model_summary_tokens["input_tokens"] += int(entry.get("input_tokens", 0))
            model_summary_tokens["output_tokens"] += int(entry.get("output_tokens", 0))
            model_summary_tokens["cached_input_tokens"] += int(entry.get("cached_input_tokens", 0))
            model_summary_tokens["reasoning_tokens"] += int(entry.get("reasoning_tokens", 0))
            model_summary_tokens["total_tokens"] += int(entry.get("total_tokens", 0))

            cost_usd = float(entry.get("cost_usd", 0.0))
            model_summary["cost_usd"] += cost_usd
            total_cost += cost_usd

            exec_time = entry.get("execution_time_seconds")
            if exec_time is not None:
                exec_time = float(exec_time)
                model_summary["timing"]["total_time_seconds"] += exec_time
                model_summary["timing"]["num_requests"] += 1
                total_time += exec_time

            if entry.get("system_message") or entry.get("prompt") or entry.get("response"):
                model_summary["interactions_count"] += 1

        for model_summary in per_model.values():
            num_requests = model_summary["timing"]["num_requests"]
            if num_requests:
                model_summary["timing"]["average_time_seconds"] = (
                    model_summary["timing"]["total_time_seconds"] / num_requests
                )
            else:
                model_summary["timing"]["average_time_seconds"] = 0.0
            total_interactions += model_summary["interactions_count"]

        summary: Dict[str, Any] = dict(per_model)
        summary["total_cost_usd"] = total_cost
        summary["total_time_seconds"] = total_time
        summary["total_entries"] = total_entries
        summary["total_interactions"] = total_interactions

        return summary

    def _save_to_file(self, new_entries: List[Dict[str, Any]]):
        """Persist cost tracking data, merging across processes safely."""
        if not self.log_file or not new_entries:
            return

        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_file, "a+", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                try:
                    f.seek(0)
                    content = f.read()
                    if content.strip():
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            logger.error("AI-Scientist cost log corrupted; recreating file")
                            data = {}
                    else:
                        data = {}

                    history = data.get("history")
                    if not isinstance(history, list):
                        history = []

                    history.extend(new_entries)
                    data["history"] = history
                    data["summary"] = self._compute_summary_from_history(history)
                    data["last_updated"] = datetime.now().isoformat()

                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=2)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to save AI-Scientist cost tracking data: {e}")

    def reset(self):
        """Reset all tracking data."""
        self.token_counts.clear()
        self.cost_history.clear()
        self.interactions.clear()
        self.timing_data.clear()
        self.instance_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._entry_sequence = 0
        if self.log_file and self.log_file.exists():
            self.log_file.unlink()


# Global cost tracker instance
ai_scientist_cost_tracker = AIScientistCostTracker()


def track_ai_scientist_usage(func):
    """
    Decorator to track token usage and costs for AI-Scientist LLM calls.
    Compatible with both sync and async functions.
    """
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract context for tracking - try multiple sources
        system_message = kwargs.get("system_message")
        prompt = kwargs.get("prompt")
        
        # If not in direct kwargs, try to extract from messages
        if not system_message or not prompt:
            messages = kwargs.get("messages", [])
            if messages and isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        if msg.get("role") == "system" and not system_message:
                            system_message = msg.get("content", "")
                        elif msg.get("role") == "user" and not prompt:
                            prompt = msg.get("content", "")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract usage information from result
            if hasattr(result, "usage") and result.usage:
                usage = result.usage
                model = getattr(result, "model", "unknown")
                
                # Extract tokens with proper handling of cached tokens
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
                
                # Handle cached tokens
                cached_tokens = 0
                if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
                
                # Handle reasoning tokens (for O1 models)
                reasoning_tokens = 0
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
                
                # Extract response content safely
                response_content = None
                if hasattr(result, "choices") and result.choices:
                    choice = result.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        response_content = choice.message.content
                
                # Track the usage
                ai_scientist_cost_tracker.add_usage(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    execution_time=execution_time,
                    system_message=system_message,
                    prompt=str(prompt) if prompt else None,
                    response=response_content,
                )
                
                # Log successful tracking
                logger.info(f"AI-Scientist tracked: {model} - {input_tokens} input, {output_tokens} output tokens")
            else:
                logger.warning(f"No usage information in response from function {func.__name__}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in AI-Scientist usage tracking: {e}")
            raise
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract context for tracking
        system_message = kwargs.get("system_message")
        prompt = kwargs.get("prompt")
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract usage information from result
            if hasattr(result, "usage") and result.usage:
                usage = result.usage
                model = getattr(result, "model", "unknown")
                
                # Extract tokens with proper handling of cached tokens
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
                
                # Handle cached tokens
                cached_tokens = 0
                if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
                
                # Handle reasoning tokens (for O1 models)
                reasoning_tokens = 0
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
                
                # Extract response content
                response_content = None
                if hasattr(result, "choices") and result.choices:
                    choice = result.choices[0]
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        response_content = choice.message.content
                
                # Track the usage
                ai_scientist_cost_tracker.add_usage(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    execution_time=execution_time,
                    system_message=system_message,
                    prompt=str(prompt) if prompt else None,
                    response=response_content,
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Error in AI-Scientist usage tracking: {e}")
            raise
    
        # Return appropriate wrapper based on function type
    import asyncio
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Global cost tracker instance
ai_scientist_cost_tracker = AIScientistCostTracker()