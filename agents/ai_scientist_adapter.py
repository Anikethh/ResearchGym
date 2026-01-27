from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import atexit
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger, CostTracker


@dataclass
class AIScientistConfig:
    task_id: str
    desc_file: Path
    steps: int
    num_workers: int
    code_model: str
    feedback_model: str
    vlm_model: str
    code_temp: float
    feedback_temp: float
    vlm_temp: float
    base_url: str = ""
    time_limit_secs: int = 3 * 60 * 60  # Allow up to three hours per run
    report_model: str = ""
    report_temp: float = 1.0
    cost_limit: float = 10.0  # Set to 0 for no limit


class AIScientistAdapter:
    def __init__(self, env: AgenticEnv, run_group: str, run_id: str) -> None:
        self.env = env
        self.run_group = run_group
        self.run_id = run_id
        self.run_logger = setup_file_logger(
            name=f"ai-scientist-agent:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )
        self.logger = setup_file_logger(
            name=f"ai-scientist-adapter:{run_id}",
            log_file=self.env.logs_dir / "adapter.log",
        )
        self.costs = CostTracker(self.env.run_dir / "costs.json")
        
        # Initialize AI-Scientist enhanced cost tracker
        self._init_ai_scientist_cost_tracker()
        
        self.proxy_process = None
        self.proxy_port = 8001
        atexit.register(self.cleanup_proxy)

    def _init_ai_scientist_cost_tracker(self):
        """Initialize the enhanced AI-Scientist cost tracker."""
        try:
            # Import and initialize the AI-Scientist cost tracker
            ai_scientist_path = Path(__file__).parent / "AI-Scientist-v2"
            sys.path.insert(0, str(ai_scientist_path))
            
            from ai_scientist.utils.cost_tracker import ai_scientist_cost_tracker
            
            # Set the log file for the AI-Scientist cost tracker
            ai_scientist_cost_tracker.log_file = self.env.run_dir / "ai_scientist_costs.json"
            
            # Reset the tracker for fresh start
            ai_scientist_cost_tracker.reset()
            
            self.logger.info("AI-Scientist enhanced cost tracker initialized")
            
        except ImportError as e:
            self.logger.warning(f"Could not initialize AI-Scientist cost tracker: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing AI-Scientist cost tracker: {e}")

    
    def prepare_workspace(self, task_dir: Path) -> None:
        input_dir = self.env.workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        for root, dirs, files in os.walk(task_dir):
            rel_path = os.path.relpath(root, task_dir)
            dst_dir = input_dir / rel_path if rel_path != "." else input_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                # Skip task-level requirements to avoid redundant installs in agent workspace
                if file.lower() in {"requirements.txt", "idea_hint.txt"}:
                    continue
                src_file = Path(root) / file
                dst_file = dst_dir / file
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    self.logger.warning(f"Failed to copy {src_file} -> {dst_file}: {e}")
    def _write_bfts_config(self, cfg: AIScientistConfig, ai_root: Path) -> Path:
        # Base config: clone template then override fields
        template = ai_root / "bfts_config.yaml"
        run_cfg = self.env.run_dir / "ai_scientist.bfts_config.yaml"
        import yaml
        base = {}
        if template.exists():
            base = yaml.safe_load(template.read_text()) or {}

        # Default dirs
        base["data_dir"] = str(self.env.workspace_dir / "input")
        base["desc_file"] = str(cfg.desc_file)
        base["workspace_dir"] = str(self.env.run_dir / "ai-sci-workspaces")
        base["log_dir"] = str(self.env.run_dir / "ai-sci-logs")
        base["copy_data"] = True
        base["generate_report"] = False
        # Agent params
        base.setdefault("agent", {})
        base["agent"]["steps"] = cfg.steps
        base["agent"]["num_workers"] = cfg.num_workers
        base["agent"].setdefault("code", {})
        base["agent"]["code"]["model"] = self._normalize_model(cfg.code_model)
        base["agent"]["code"]["temp"] = cfg.code_temp
        base["agent"].setdefault("feedback", {})
        base["agent"]["feedback"]["model"] = self._normalize_model(cfg.feedback_model)
        base["agent"]["feedback"]["temp"] = cfg.feedback_temp
        base["agent"].setdefault("vlm_feedback", {})
        base["agent"]["vlm_feedback"]["model"] = self._normalize_model(cfg.vlm_model)
        base["agent"]["vlm_feedback"]["temp"] = cfg.vlm_temp

        # Report params
        base.setdefault("report", {})
        base["report"]["model"] = self._normalize_model(cfg.report_model)
        base["report"]["temp"] = cfg.report_temp

        # Exec params
        base.setdefault("exec", {})
        base["exec"]["timeout"] = cfg.time_limit_secs
        base["exec"]["agent_file_name"] = "runfile.py"
        base["exec"]["format_tb_ipython"] = False

        # Write
        run_cfg.write_text(yaml.dump(base))
        return run_cfg

    @staticmethod
    def _is_wsl_stub(path: str) -> bool:
        lowered = path.lower()
        return (
            "windowsapps\\bash.exe" in lowered
            or "system32\\bash.exe" in lowered
            or "sysnative\\bash.exe" in lowered
        )

    def _ensure_portable_bash(self, env: Dict[str, str]) -> None:
        """Ensure Windows runs bash scripts with Git Bash when available."""
        if os.name != "nt":
            return

        import shutil

        current = shutil.which("bash", path=env.get("PATH"))
        if current and not self._is_wsl_stub(current):
            env.setdefault("RG_BASH_PATH", current)
            return

        candidates = [
            env.get("RG_BASH_PATH"),
            os.environ.get("RG_BASH_PATH"),
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files\Git\bin",
            r"C:\Program Files\Git\usr\bin",
        ]

        for cand in candidates:
            if not cand:
                continue
            path_candidate = Path(cand)
            if path_candidate.is_dir():
                bash_path = path_candidate / "bash.exe"
            else:
                bash_path = path_candidate

            if bash_path.exists() and not self._is_wsl_stub(str(bash_path)):
                env["RG_BASH_PATH"] = str(bash_path)
                parent_dir = bash_path.parent
                env["PATH"] = f"{parent_dir}{os.pathsep}" + env.get("PATH", "")
                return

        # As a last resort, prefer Git's cmd folder if present (contains sh.exe)
        git_cmd = Path(r"C:\Program Files\Git\cmd")
        if git_cmd.exists():
            env["PATH"] = f"{git_cmd}{os.pathsep}" + env.get("PATH", "")

    def _normalize_model(self, model: str) -> str:
        # Keep provider/model for non-OpenAI providers so LiteLLM can route precisely
        if not model:
            return model
        if "/" not in model:
            return model
        provider, name = model.split("/", 1)
        if provider in ("openai", "azure"):
            return name
        # For gemini, remove google/ prefix if no proxy is used
        if provider == "google":
            return name
        # For others (e.g., gemini), keep provider/model
        return model

    def setup_litellm_proxy(self, models: List[str]) -> str:
        """Set up LiteLLM proxy for Gemini models with proper API key handling."""
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY not found in environment")

            self.logger.info("Using GEMINI_API_KEY from environment")
            
            import yaml
            model_list = []
            for m in models:
                if not m:
                    continue
                    
                # Handle different model formats
                if "/" in m:
                    provider, name = m.split("/", 1)
                    if provider in ["openai", "azure"]:
                        continue  # Skip OpenAI models, they don't need proxy
                else:
                    provider = "unknown"
                    name = m
                
                # Handle gemini/google models specifically
                if provider in ["google"] or "gemini" in m.lower():
                    # For Gemini models, use the correct LiteLLM format for Google AI Studio
                    # Use "gemini/" prefix for Google AI Studio API (not Vertex AI)
                    if m.startswith("gemini-"):
                        # For direct gemini models, add gemini/ prefix for AI Studio
                        litellm_model = f"gemini/{m}"
                    elif provider == "google":
                        litellm_model = f"gemini/{name}"
                    else:
                        litellm_model = f"gemini/{m}"
                    
                    params = {"model": litellm_model, "api_key": gemini_api_key}
                    model_list.append({"model_name": m, "litellm_params": params})
                    self.logger.info(f"Added Gemini model to proxy: {m} -> {litellm_model}")
                else:
                    # Handle other providers
                    api_key = os.getenv(f"{provider.upper()}_API_KEY")
                    params = {"model": name if provider != "unknown" else m}
                    if api_key:
                        params["api_key"] = api_key
                    model_list.append({"model_name": m, "litellm_params": params})

            if not model_list:
                raise RuntimeError("No models require proxy - all models can be used directly")

            config = {
                "model_list": model_list, 
                "general_settings": {
                    # "master_key": "sk1234",  # Remove master_key to disable auth - causes issues
                    "completion_proxy_timeout": 120,  # Increase timeout
                    "proxy_budget_rpm": 30000,  # Set rate limits
                }
            }
            cfg_path = self.env.run_dir / "litellm_ai_sci.yaml"
            cfg_path.write_text(yaml.dump(config))
            
            self.logger.info(f"LiteLLM config written to {cfg_path}")
            self.logger.info(f"Model list: {[m['model_name'] for m in model_list]}")

            cmd = ["litellm", "--config", str(cfg_path), "--port", str(self.proxy_port), "--num_workers", "1"]
            self.logger.info(f"Starting LiteLLM proxy: {' '.join(cmd)}")
            self.proxy_process = subprocess.Popen(
                cmd,
                stdout=open(self.env.logs_dir / "litellm_ai_sci.stdout.log", "w"),
                stderr=open(self.env.logs_dir / "litellm_ai_sci.stderr.log", "w"),
                cwd=str(self.env.run_dir),
            )
            
            # Wait longer and check if proxy started successfully
            for i in range(15):  # Check for 15 seconds
                time.sleep(1)
                if self.proxy_process.poll() is not None:
                    # Process died, read error logs
                    stderr_log = self.env.logs_dir / "litellm_ai_sci.stderr.log"
                    if stderr_log.exists():
                        error_content = stderr_log.read_text()
                        self.logger.error(f"LiteLLM proxy failed to start. Error: {error_content}")
                    raise RuntimeError("LiteLLM proxy failed to start")
                
                # Test proxy connectivity
                try:
                    import requests
                    response = requests.get(f"http://localhost:{self.proxy_port}/health", timeout=5)
                    if response.status_code == 200:
                        self.logger.info("LiteLLM proxy health check passed")
                        break
                except Exception as e:
                    self.logger.debug(f"Proxy health check attempt {i+1} failed: {e}")
                    if i == 14:  # Last attempt
                        self.logger.warning(f"LiteLLM proxy health check failed after 15 attempts, but proceeding anyway")
            
            return f"http://localhost:{self.proxy_port}/v1"
            
        except Exception as e:
            self.logger.error(f"LiteLLM proxy setup failed: {e}")
            # Clean up failed proxy process
            if self.proxy_process:
                try:
                    self.proxy_process.terminate()
                    self.proxy_process.wait(timeout=5)
                except:
                    try:
                        self.proxy_process.kill()
                    except:
                        pass
                self.proxy_process = None
            raise

    def cleanup_proxy(self):
        if self.proxy_process:
            try:
                self.proxy_process.terminate()
                self.proxy_process.wait(timeout=10)
            except Exception:
                try:
                    self.proxy_process.kill()
                except Exception:
                    pass
            finally:
                self.proxy_process = None

    def build_command(self, cfg: AIScientistConfig, ai_root: Path) -> list[str]:
        # Ensure desc_file exists; it can be MD or JSON; config layer will wrap MD
        run_cfg = self._write_bfts_config(cfg, ai_root)
        python = sys.executable
        entry = ai_root / "ai_scientist" / "treesearch" / "perform_experiments_bfts_with_agentmanager.py"
        # We invoke as a module script: python -c 'from ... import perform_experiments_bfts; perform_experiments_bfts(...)'
        code = (
            "import sys; sys.path.insert(0, " + repr(str(ai_root)) + "); "
            "from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager "
            "import perform_experiments_bfts; perform_experiments_bfts(" + repr(str(run_cfg)) + ")"
        )
        cmd = [python, "-c", code]
        
        # Environment setup
        env_hint = {}
        
        # Optional LiteLLM/OpenAI-compatible proxy/base URL
        if cfg.base_url:
            # Use provided proxy
            env_hint["OPENAI_BASE_URL"] = cfg.base_url
            env_hint["OPENAI_API_KEY"] = "dummy-key"  # Some dummy key for proxy
        else:
            # Direct API usage
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                env_hint["GEMINI_API_KEY"] = gemini_key
                    
        # Add timeout and backoff settings to prevent infinite retries
        env_hint.update({
            "OPENAI_MAX_RETRIES": "5",  # Limit retries
            "OPENAI_TIMEOUT": "60",     # Set reasonable timeout
        })
        
        # Add cost limit similar to BasicAgent
        cost_limit = getattr(cfg, 'cost_limit', 10.0)
        if cost_limit > 0:
            env_hint["AI_SCIENTIST_COST_LIMIT"] = str(cost_limit)
            self.logger.info(f"Setting AI-Scientist cost limit: ${cost_limit:.2f}")
        
        # Set model environment variables to override hardcoded models
        # Use the feedback model for summary/selection tasks as fallback
        summary_model = cfg.feedback_model or cfg.code_model
        if summary_model:
            env_hint["AI_SCIENTIST_SUMMARY_MODEL"] = self._normalize_model(summary_model)
            env_hint["AI_SCIENTIST_SELECTION_MODEL"] = self._normalize_model(summary_model)
            self.logger.info(f"Set AI-Scientist dynamic models: {self._normalize_model(summary_model)}")

        # Save hints for run()
        self._last_env_hint = env_hint
        return cmd

    def run(self, cfg: AIScientistConfig, ai_root: Path, env_vars: Optional[Dict[str, str]] = None) -> Observation:
        cmd = self.build_command(cfg, ai_root)
        self.logger.info(f"AI-Scientist command: {' '.join(cmd)}")
        self.run_logger.info(f"starting: {' '.join(cmd)}")

        env = os.environ.copy()
        if hasattr(self, "_last_env_hint"):
            env.update(getattr(self, "_last_env_hint"))
        if env_vars:
            env.update(env_vars)

        self._ensure_portable_bash(env)

        proc = subprocess.Popen(
            cmd,
            cwd=str(ai_root),
            env=env,
            stdout=open(self.env.logs_dir / "ai_scientist.stdout.log", "w"),
            stderr=open(self.env.logs_dir / "ai_scientist.stderr.log", "w"),
        )
        rc = proc.wait(timeout=cfg.time_limit_secs)
        
        # After execution, get cost summary from AI-Scientist tracker
        cost_summary = self._get_ai_scientist_cost_summary()
        
        # Integrate AI-Scientist costs into ResearchGym's cost tracker
        self._integrate_ai_scientist_costs(cost_summary)
        
        return Observation(
            message="completed", 
            info={
                "returncode": rc,
                "ai_scientist_costs": cost_summary
            }
        )

    def _integrate_ai_scientist_costs(self, cost_summary: Dict):
        """Integrate AI-Scientist costs into ResearchGym's cost tracker."""
        try:
            total_cost = cost_summary.get("total_cost_usd", 0.0)
            total_time = cost_summary.get("total_time_seconds", 0.0) 
            total_requests = cost_summary.get("total_entries", 0)
            total_interactions = cost_summary.get("total_interactions", 0)
            
            # Add model-by-model data to ResearchGym's cost tracker
            for model_name, model_data in cost_summary.items():
                if isinstance(model_data, dict) and "tokens" in model_data:
                    tokens = model_data["tokens"]
                    input_tokens = tokens.get("input_tokens", 0)
                    output_tokens = tokens.get("output_tokens", 0)
                    
                    # Add to ResearchGym's cost tracker
                    self.costs.add(model_name, input_tokens, output_tokens)
            
            self.logger.info(f"Integrated AI-Scientist costs into ResearchGym tracker: ${total_cost:.6f}")
            
        except Exception as e:
            self.logger.error(f"Failed to integrate AI-Scientist costs: {e}")

    def _get_ai_scientist_cost_summary(self) -> Dict:
        """Get cost summary from AI-Scientist cost tracker."""
        try:
            # First, try to parse cost summary from exec.stdout.log
            exec_log_path = self.env.run_dir / "logs" / "exec.stdout.log"
            
            if exec_log_path.exists():
                # Import our cost parser
                from ResearchGym.utils.cost_summary_parser import parse_cost_summary_from_log, format_cost_summary
                
                summary = parse_cost_summary_from_log(exec_log_path)
                
                if "error" not in summary:
                    # Save the parsed summary
                    cost_summary_path = exec_log_path.parent / "ai_scientist_cost_summary.json"
                    with open(cost_summary_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    # Create formatted text summary
                    formatted_summary = format_cost_summary(summary)
                    summary_txt_path = exec_log_path.parent / "ai_scientist_cost_summary.txt"
                    with open(summary_txt_path, 'w') as f:
                        f.write(formatted_summary)
                    
                    self.logger.info(f"Generated AI-Scientist cost summary:")
                    for line in formatted_summary.split('\n'):
                        if line.strip():
                            self.logger.info(line)
                    
                    self.logger.info(f"Cost summary saved to: {cost_summary_path}")
                    self.logger.info(f"Formatted summary saved to: {summary_txt_path}")
                    
                    return summary
                else:
                    self.logger.warning(f"Failed to parse cost summary from log: {summary['error']}")
            
            # Prefer vendor's saved summary under ai-sci-logs/<exp_name>/
            try:
                base_dir = self.env.run_dir / "ai-sci-logs"
                if base_dir.exists():
                    candidates = list(base_dir.rglob("ai_scientist_cost_summary.json"))
                    if candidates:
                        latest = sorted(candidates)[-1]
                        with open(latest, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        self.logger.info(f"Loaded AI-Scientist cost summary from file: {latest}")
                        return summary
            except Exception as e:
                self.logger.warning(f"Failed to load vendor cost summary file: {e}")
            
            # Last resort: try to access the cost tracker directly
            ai_scientist_path = Path(__file__).parent / "AI-Scientist-v2"
            sys.path.insert(0, str(ai_scientist_path))
            
            from ai_scientist.utils.cost_tracker import ai_scientist_cost_tracker
            summary = ai_scientist_cost_tracker.get_summary()
            self.logger.info("Loaded AI-Scientist cost summary from memory")
            
            # Log the summary
            total_cost = summary.get("total_cost_usd", 0.0)
            total_time = summary.get("total_time_seconds", 0.0)
            total_requests = summary.get("total_entries", 0)
            
            self.logger.info(f"AI-Scientist execution completed:")
            self.logger.info(f"  Total cost: ${total_cost:.6f}")
            self.logger.info(f"  Total time: {total_time:.2f} seconds") 
            self.logger.info(f"  Total requests: {total_requests}")
            
            # Log per-model breakdown
            for model, model_data in summary.items():
                if isinstance(model_data, dict) and "cost_usd" in model_data:
                    tokens = model_data.get("tokens", {})
                    timing = model_data.get("timing", {})
                    self.logger.info(f"  {model}: ${model_data['cost_usd']:.6f} "
                                   f"({tokens.get('total_tokens', 0)} tokens, "
                                   f"{timing.get('num_requests', 0)} requests)")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get AI-Scientist cost summary: {e}")
            return {
                "total_cost_usd": 0.0,
                "total_time_seconds": 0.0,
                "total_entries": 0,
                "error": str(e)
            }


