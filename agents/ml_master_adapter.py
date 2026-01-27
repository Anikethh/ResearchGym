from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import yaml
import time
import signal
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger, CostTracker


@dataclass
class MLMasterConfig:
    # Minimal subset to launch ML-Master against a ResearchGym task folder
    task_id: str
    desc_file: Optional[Path]
    code_model: str
    code_temp: float
    code_base_url: str
    code_api_key: str
    feedback_model: str
    feedback_temp: float
    feedback_base_url: str
    feedback_api_key: str
    time_limit_secs: int = 3600


class MLMasterAdapter:
    def __init__(self, env: AgenticEnv, run_group: str, run_id: str) -> None:
        self.env = env
        self.run_group = run_group
        self.run_id = run_id
        # run-level agent log at root
        self.run_logger = setup_file_logger(
            name=f"ml-master-agent:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )
        self.logger = setup_file_logger(
            name=f"ml-master-adapter:{run_id}",
            log_file=self.env.logs_dir / "adapter.log",
        )
        self.costs = CostTracker(self.env.run_dir / "costs.json")
        self.proxy_process = None
        self.proxy_port = 8000
        
        # Register cleanup handler
        atexit.register(self.cleanup_proxy)

    def prepare_workspace(self, task_dir: Path) -> None:
        # Mirror mle-bench style: create input/public with description and data
        input_dir = self.env.workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        # For now, copy the task repo into workspace/input; keep it simple & safe
        if (self.env.workspace_dir / "input").exists():
            # copy tree, ignoring heavy dirs like .git and task requirements.txt
            shutil.copytree(
                task_dir,
                input_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "requirements.txt", "idea_hint.txt"),
            )

    def build_mlmaster_command(self, cfg: MLMasterConfig, ml_master_root: Path) -> list[str]:
        # For ResearchGym: input dir contains task files directly (not MLE-Bench structure)
        # We set both dataset_dir (for schema validation) and data_dir (actual working directory)
        input_dir = self.env.workspace_dir / "input"
        logs_root = self.env.run_dir / "ml-master-logs"
        work_root = self.env.run_dir / "ml-master-workspaces"
        args: list[str] = [
            sys.executable,
            str(ml_master_root / "main_mcts.py"),
            f"dataset_dir={input_dir}",  # Satisfies schema validation
            f"data_dir={input_dir}",     # Direct working directory for ResearchGym tasks
            f"log_dir={logs_root}",
            f"workspace_dir={work_root}",
            f"exp_name={cfg.task_id}_rg",
            f"agent.code.model={cfg.code_model}",
            f"agent.code.temp={cfg.code_temp}",
            f"agent.code.base_url={cfg.code_base_url}",
            f"agent.code.api_key={cfg.code_api_key}",
            f"agent.feedback.model={cfg.feedback_model}",
            f"agent.feedback.temp={cfg.feedback_temp}",
            f"agent.feedback.base_url={cfg.feedback_base_url}",
            f"agent.feedback.api_key={cfg.feedback_api_key}",
            f"start_cpu_id=0",
            f"cpu_number=3",  # Match the default parallel_search_num of 3
        ]
        # Prefer description.md inside data_dir, else fallback to provided desc_file, else goal string
        data_desc = input_dir / "task_description.md"
        if data_desc.exists():
            args.append(f'desc_file="{data_desc}"')
        elif cfg.desc_file and Path(cfg.desc_file).exists():
            args.append(f'desc_file="{cfg.desc_file}"')
        else:
            # Quote value to prevent Hydra from interpreting colon as a dict
            args.append(f'goal="Solve ResearchGym task: {cfg.task_id}"')
        return args

    def setup_litellm_proxy(self, code_model: str, feedback_model: str) -> str:
        """Setup a generic LiteLLM proxy for any models that need OpenAI-compatible endpoints."""
        try:
            # Extract unique models and their configurations
            models_to_configure = []
            
            # Helper function to extract model info and add to config
            def add_model_config(model_name: str):
                if not model_name or "/" not in model_name:
                    return  # Skip if not in provider/model format
                
                provider = model_name.split("/")[0]
                
                # Skip if it's already OpenAI compatible
                if provider in ["openai", "azure"]:
                    return
                
                # Add to models list if not already there
                if not any(m["model_name"] == model_name for m in models_to_configure):
                    # Get API key based on provider
                    api_key = None
                    if provider == "gemini":
                        api_key = os.getenv("GEMINI_API_KEY")
                    elif provider == "anthropic":
                        api_key = os.getenv("ANTHROPIC_API_KEY")
                    elif provider == "cohere":
                        api_key = os.getenv("COHERE_API_KEY")
                    elif provider == "mistral":
                        api_key = os.getenv("MISTRAL_API_KEY")
                    # Add more providers as needed
                    
                    litellm_params = {"model": model_name}
                    if api_key:
                        litellm_params["api_key"] = api_key
                    
                    models_to_configure.append({
                        "model_name": model_name,
                        "litellm_params": litellm_params
                    })
            
            add_model_config(code_model)
            add_model_config(feedback_model)
            
            if not models_to_configure:
                raise ValueError("No models requiring LiteLLM proxy found")
            
            # Create generic LiteLLM config
            config = {
                "model_list": models_to_configure,
                "general_settings": {
                    "master_key": "sk-1234"
                }
            }
            
            # Write config to run directory
            config_path = self.env.run_dir / "litellm_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Start LiteLLM proxy server
            cmd = ["litellm", "--config", str(config_path), "--port", str(self.proxy_port)]
            self.logger.info(f"Starting LiteLLM proxy: {' '.join(cmd)}")
            
            # Start the proxy process
            self.proxy_process = subprocess.Popen(
                cmd,
                stdout=open(self.env.logs_dir / "litellm_proxy.stdout.log", "w"),
                stderr=open(self.env.logs_dir / "litellm_proxy.stderr.log", "w"),
                cwd=str(self.env.run_dir)
            )
            
            # Wait a bit for the server to start
            time.sleep(5)
            
            # Check if process is still running
            if self.proxy_process.poll() is not None:
                raise RuntimeError("LiteLLM proxy failed to start")
            
            proxy_url = f"http://localhost:{self.proxy_port}/v1"
            self.logger.info(f"LiteLLM proxy started at {proxy_url}")
            return proxy_url
            
        except Exception as e:
            self.logger.error(f"Failed to setup LiteLLM proxy: {e}")
            raise RuntimeError(f"Failed to setup LiteLLM proxy: {e}")
    
    def cleanup_proxy(self):
        """Cleanup the LiteLLM proxy process."""
        if self.proxy_process:
            try:
                self.logger.info("Terminating LiteLLM proxy")
                self.proxy_process.terminate()
                self.proxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("LiteLLM proxy didn't terminate gracefully, killing")
                self.proxy_process.kill()
            except Exception as e:
                self.logger.error(f"Error cleaning up proxy: {e}")
            finally:
                self.proxy_process = None

    def run(self, cfg: MLMasterConfig, ml_master_root: Path, env_vars: Optional[Dict[str, str]] = None, dry_run: bool = True) -> Observation:
        cmd = self.build_mlmaster_command(cfg, ml_master_root)
        self.logger.info(f"ML-Master command: {' '.join(cmd)}")
        self.run_logger.info(f"planning: {' '.join(cmd)}")
        if dry_run:
            return Observation(message="planned", info={"command": cmd})

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        proc = subprocess.Popen(
            cmd,
            cwd=str(ml_master_root),
            env=env,
            stdout=open(self.env.logs_dir / "ml_master.stdout.log", "w"),
            stderr=open(self.env.logs_dir / "ml_master.stderr.log", "w"),
        )
        rc = proc.wait(timeout=cfg.time_limit_secs)
        return Observation(message="completed", info={"returncode": rc})


