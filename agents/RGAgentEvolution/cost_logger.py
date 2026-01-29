"""
Cost logging utilities for RGAgentEvolution.
Provides functions to extract costs from existing logs and add cost information.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def calculate_cost_from_usage(model: str, usage_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate cost from existing usage data.
    
    Args:
        model: Model name
        usage_data: Usage data dictionary with token counts
        
    Returns:
        Dictionary with cost information
    """
    tracker = CostTracker()
    
    input_tokens = usage_data.get('input_tokens', 0)
    output_tokens = usage_data.get('output_tokens', 0)
    cached_tokens = usage_data.get('input_tokens_cache_read', 0)
    reasoning_tokens = usage_data.get('reasoning_tokens', 0)
    
    cost = tracker._calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    
    breakdown = tracker._get_cost_breakdown(model, {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'cached_input_tokens': cached_tokens,
        'reasoning_tokens': reasoning_tokens,
    })
    
    return {
        'total_cost_usd': cost,
        'cost_breakdown': breakdown,
        'pricing_model': model,
    }


def add_cost_to_log_file(log_file_path: Path) -> None:
    """
    Add cost information to an existing RGAgentEvolution log file.
    
    Args:
        log_file_path: Path to the log file
    """
    try:
        # Read the log file
        log_data = json.loads(log_file_path.read_text())
        
        # Check if cost information already exists
        if 'cost_summary' in log_data.get('stats', {}):
            logger.info(f"Cost information already exists in {log_file_path}")
            return
        
        # Extract usage data
        stats = log_data.get('stats', {})
        model_usage = stats.get('model_usage', {})
        
        if not model_usage:
            logger.warning(f"No model usage data found in {log_file_path}")
            return
        
        # Calculate costs for each model
        total_cost = 0.0
        cost_details = {}
        
        for model, usage in model_usage.items():
            cost_info = calculate_cost_from_usage(model, usage)
            cost_details[model] = cost_info
            total_cost += cost_info['total_cost_usd']
        
        # Add cost summary to stats
        stats['cost_summary'] = {
            'total_cost_usd': total_cost,
            'cost_by_model': cost_details,
            'cost_calculated_at': CostTracker().cost_history[-1]['timestamp'] if CostTracker().cost_history else None,
        }
        
        # Save updated log
        log_file_path.write_text(json.dumps(log_data, indent=2))
        logger.info(f"Added cost information to {log_file_path}: ${total_cost:.6f}")
        
    except Exception as e:
        logger.error(f"Failed to add cost information to {log_file_path}: {e}")


def add_cost_to_all_logs(logs_dir: Path) -> None:
    """
    Add cost information to all log files in a directory.
    
    Args:
        logs_dir: Directory containing log files
    """
    if not logs_dir.exists():
        logger.error(f"Logs directory does not exist: {logs_dir}")
        return
    
    log_files = list(logs_dir.glob("*.json"))
    if not log_files:
        logger.warning(f"No JSON log files found in {logs_dir}")
        return
    
    logger.info(f"Processing {len(log_files)} log files...")
    
    total_cost = 0.0
    for log_file in log_files:
        try:
            add_cost_to_log_file(log_file)
            
            # Read back to get total cost
            log_data = json.loads(log_file.read_text())
            file_cost = log_data.get('stats', {}).get('cost_summary', {}).get('total_cost_usd', 0.0)
            total_cost += file_cost
            
        except Exception as e:
            logger.error(f"Failed to process {log_file}: {e}")
    
    logger.info(f"Cost processing complete. Total cost across all files: ${total_cost:.6f}")


def print_cost_summary(logs_dir: Path) -> None:
    """
    Print a summary of costs from all log files.
    
    Args:
        logs_dir: Directory containing log files
    """
    if not logs_dir.exists():
        print(f"Logs directory does not exist: {logs_dir}")
        return
    
    log_files = list(logs_dir.glob("*.json"))
    if not log_files:
        print(f"No JSON log files found in {logs_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"RGAGENT EVOLUTION COST SUMMARY")
    print(f"{'='*60}")
    
    total_cost = 0.0
    model_totals = {}
    
    for log_file in log_files:
        try:
            log_data = json.loads(log_file.read_text())
            stats = log_data.get('stats', {})
            cost_summary = stats.get('cost_summary', {})
            
            if not cost_summary:
                continue
                
            file_cost = cost_summary.get('total_cost_usd', 0.0)
            total_cost += file_cost
            
            print(f"\nFile: {log_file.name}")
            print(f"Cost: ${file_cost:.6f}")
            
            # Show model breakdown
            cost_by_model = cost_summary.get('cost_by_model', {})
            for model, cost_info in cost_by_model.items():
                model_cost = cost_info.get('total_cost_usd', 0.0)
                model_totals[model] = model_totals.get(model, 0.0) + model_cost
                print(f"  {model}: ${model_cost:.6f}")
            
        except Exception as e:
            print(f"Error processing {log_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"TOTALS")
    print(f"{'='*60}")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"\nBy Model:")
    for model, cost in model_totals.items():
        print(f"  {model}: ${cost:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    logs_dir = Path(__file__).parent / "logs"
    
    # Add cost information to all logs
    add_cost_to_all_logs(logs_dir)
    
    # Print summary
    print_cost_summary(logs_dir)
