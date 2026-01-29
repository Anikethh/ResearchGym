"""
Cost tracking module for RGAgent with support for cached tokens.
Tracks and calculates costs for GPT-5 and Gemini models.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CostTracker:
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
        self.log_file = log_file
        
        # Model pricing (per 1M tokens)
        self.MODEL_PRICES = {
            # GPT-5 pricing
            "gpt-5": {
                "input": 1.25,  # $1.25 per 1M tokens
                "cached_input": 0.125,  # $0.125 per 1M tokens
                "output": 10.00,  # $10.00 per 1M tokens
            },
            
            # Gemini 2.5 Pro pricing (tiered)
            "google/gemini-2.5-pro": {
                "input_low": 1.25,    # $1.25 per 1M tokens (≤200k)
                "input_high": 2.50,   # $2.50 per 1M tokens (>200k)
                "output_low": 10.00,  # $10.00 per 1M tokens (≤200k)
                "output_high": 15.00, # $15.00 per 1M tokens (>200k)
                "cached_low": 0.31,   # $0.31 per 1M tokens (≤200k)
                "cached_high": 0.625, # $0.625 per 1M tokens (>200k)
                "threshold": 200000,  # 200k token threshold
            },
            
            # Gemini 2.5 Flash Lite pricing
            "google/gemini-2.5-flash-lite": {
                "input": 0.10,   # $0.10 per 1M tokens
                "output": 0.40,  # $0.40 per 1M tokens
                "cached_input": 0.025,  # $0.025 per 1M tokens
            },

            # Gemini 3.0 Pro (preview) pricing (tiered)
            "google/gemini-3-pro-preview": {
                "input_low": 2.00,     # $2.00 per 1M tokens (≤200k)
                "input_high": 4.00,    # $4.00 per 1M tokens (>200k)
                "output_low": 12.00,   # $12.00 per 1M tokens (≤200k)
                "output_high": 18.00,  # $18.00 per 1M tokens (>200k)
                "cached_low": 0.20,    # $0.20 per 1M tokens (≤200k)
                "cached_high": 0.40,   # $0.40 per 1M tokens (>200k)
                "threshold": 200000,   # 200k token threshold
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
            
        Returns:
            Cost for this usage in USD
        """
        # Update counts
        self.token_counts[model]["input_tokens"] += input_tokens
        self.token_counts[model]["output_tokens"] += output_tokens
        self.token_counts[model]["cached_input_tokens"] += cached_input_tokens
        self.token_counts[model]["reasoning_tokens"] += reasoning_tokens
        
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens + cached_input_tokens + reasoning_tokens
        self.token_counts[model]["total_tokens"] += total_tokens
        
        # Calculate cost
        cost = self._calculate_cost(model, input_tokens, output_tokens, cached_input_tokens, reasoning_tokens)
        
        # Log this usage
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
        }
        self.cost_history.append(usage_entry)
        
        # Save to file if configured
        if self.log_file:
            self._save_to_file()
            
        logger.info(f"Cost tracking - Model: {model}, Cost: ${cost:.6f}, "
                   f"Input: {input_tokens}, Cached: {cached_input_tokens}, Output: {output_tokens}")
        
        return cost

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> float:
        """Calculate cost for given usage."""
        normalized_model = model
        billed_output_tokens = output_tokens + reasoning_tokens
        if "gemini-2.5-pro" in model:
            normalized_model = "google/gemini-2.5-pro"
        elif "gemini-2.5-flash-lite" in model:
            normalized_model = "google/gemini-2.5-flash-lite"
        elif "gemini-3" in model and "pro" in model:
            normalized_model = "google/gemini-3-pro-preview"

        if normalized_model not in self.MODEL_PRICES:
            logger.warning(f"No pricing information for model: {model}")
            return 0.0
            
        prices = self.MODEL_PRICES[normalized_model]
        cost = 0.0
        
        # Handle tiered pricing for Gemini 2.5 Pro and Gemini 3 Pro Preview
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
        if model:
            tokens = self.token_counts[model]
            return self._calculate_cost(
                model,
                tokens["input_tokens"],
                tokens["output_tokens"],
                tokens["cached_input_tokens"],
                tokens["reasoning_tokens"],
            )
        else:
            return sum(
                self._calculate_cost(
                    model_name,
                    tokens["input_tokens"],
                    tokens["output_tokens"],
                    tokens["cached_input_tokens"],
                    tokens["reasoning_tokens"],
                )
                for model_name, tokens in self.token_counts.items()
            )

    def get_summary(self) -> Dict:
        """Get comprehensive summary of usage and costs."""
        summary = {}
        total_cost = 0.0
        
        for model, tokens in self.token_counts.items():
            model_cost = self._calculate_cost(
                model,
                tokens["input_tokens"],
                tokens["output_tokens"],
                tokens["cached_input_tokens"],
                tokens["reasoning_tokens"],
            )
            total_cost += model_cost
            
            summary[model] = {
                "tokens": tokens.copy(),
                "cost_usd": model_cost,
                "cost_breakdown": self._get_cost_breakdown(model, tokens),
            }
        
        summary["total_cost_usd"] = total_cost
        summary["total_entries"] = len(self.cost_history)
        
        return summary

    def _get_cost_breakdown(self, model: str, tokens: Dict) -> Dict:
        """Get detailed cost breakdown for a model."""
        normalized_model = model
        if "gemini-2.5-pro" in model:
            normalized_model = "google/gemini-2.5-pro"
        elif "gemini-2.5-flash-lite" in model:
            normalized_model = "google/gemini-2.5-flash-lite"
        elif "gemini-3" in model and "pro" in model:
            normalized_model = "google/gemini-3-pro-preview"

        if normalized_model not in self.MODEL_PRICES:
            return {}
            
        prices = self.MODEL_PRICES[normalized_model]
        breakdown = {}
        
        if normalized_model in ("google/gemini-2.5-pro", "google/gemini-3-pro-preview"):
            threshold = prices["threshold"]
            input_tokens = tokens["input_tokens"]
            output_tokens = tokens["output_tokens"]
            cached_tokens = tokens["cached_input_tokens"]
            
            # Input cost breakdown
            if input_tokens <= threshold:
                breakdown["input_cost"] = input_tokens * prices["input_low"] / 1_000_000
            else:
                low_cost = threshold * prices["input_low"] / 1_000_000
                high_cost = (input_tokens - threshold) * prices["input_high"] / 1_000_000
                breakdown["input_cost"] = low_cost + high_cost
                breakdown["input_cost_low_tier"] = low_cost
                breakdown["input_cost_high_tier"] = high_cost
            
            # Output cost breakdown
            if output_tokens <= threshold:
                breakdown["output_cost"] = output_tokens * prices["output_low"] / 1_000_000
            else:
                low_cost = threshold * prices["output_low"] / 1_000_000
                high_cost = (output_tokens - threshold) * prices["output_high"] / 1_000_000
                breakdown["output_cost"] = low_cost + high_cost
                breakdown["output_cost_low_tier"] = low_cost
                breakdown["output_cost_high_tier"] = high_cost
            
            # Cached cost breakdown
            if cached_tokens <= threshold:
                breakdown["cached_cost"] = cached_tokens * prices["cached_low"] / 1_000_000
            else:
                low_cost = threshold * prices["cached_low"] / 1_000_000
                high_cost = (cached_tokens - threshold) * prices["cached_high"] / 1_000_000
                breakdown["cached_cost"] = low_cost + high_cost
                breakdown["cached_cost_low_tier"] = low_cost
                breakdown["cached_cost_high_tier"] = high_cost
        else:
            # Flat pricing breakdown
            if "input" in prices:
                breakdown["input_cost"] = tokens["input_tokens"] * prices["input"] / 1_000_000
            if "cached_input" in prices:
                breakdown["cached_cost"] = tokens["cached_input_tokens"] * prices["cached_input"] / 1_000_000
            if "output" in prices:
                breakdown["output_cost"] = tokens["output_tokens"] * prices["output"] / 1_000_000
        
        return breakdown

    def _save_to_file(self):
        """Save cost tracking data to file."""
        if not self.log_file:
            return
            
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "summary": self.get_summary(),
                "history": self.cost_history,
                "last_updated": datetime.now().isoformat(),
            }
            self.log_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save cost tracking data: {e}")

    def reset(self):
        """Reset all tracking data."""
        self.token_counts.clear()
        self.cost_history.clear()
        if self.log_file and self.log_file.exists():
            self.log_file.unlink()


# Global cost tracker instance
cost_tracker = CostTracker()
