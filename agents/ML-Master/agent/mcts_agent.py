import shutil
import logging
import random
import os
import time
from typing import Any, Callable, cast, Tuple, List, Literal
import math
import humanize
from backend import FunctionSpec, compile_prompt_to_md, query, r1_query
from interpreter.interpreter_parallel import ExecutionResult
from search.journal import Journal
from search.mcts_node import MCTSNode
import utils.data_preview as data_preview
from utils.config_mcts import Config
from utils.metric import MetricValue, WorstMetricValue
from utils.response import extract_code, extract_text_up_to_code, wrap_code, extract_review
from utils.server_utils import call_validate
from utils.mcts import linear_decay, exponential_decay, piecewise_decay, dynamic_piecewise_decay
import threading

logger = logging.getLogger("ml-master")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class MCTSAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.journal = journal
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: MCTSNode | None = None
        self.all_root = True
        self.virtual_root = MCTSNode(parent=None, plan="virtual plan", code="# virtual code", metric=WorstMetricValue(), stage="root")
        self.current_node_list = []
        self.journal.append(self.virtual_root)
        self.best_metric: float = None
        self.best_node: MCTSNode = None
        self.search_start_time = None
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()
        
    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "transformers",
            "nltk",
            "spacy",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt
    
    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code must not only implement the proposed solution but also **print the evaluation metric computed on a hold-out validation set**. **Without this metric, the solution cannot be evaluated, rendering the entire code invalid.**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
            "If you use `DataLoader`, you need to increase the parameter `num_workers` to speed up the training process."
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}
    
    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }
    
    def _draft(self) -> MCTSNode:
        logger.info("Starting Drafting a new Node.")
        introduction = (
            "As an expert in the field of AI/ML research, having published multiple award winning papers in top-tier conferences like ICML, ICLR, CVPR and ACL, "
            "your task is to come up with an idea and implement it to tackle the stated research goal. You can utilise an existing repository as a starting point "
            "and try to improve performance over the baselines in the results table. Do not try to cheat or hack your way to improving the results. "
            "Your proposed ideas should be grounded, well motivated and novel. You can keep updating, refining or refreshing your idea until you are satisfied with the results. "
            "You can leverage feedback from executing the proposed hypotheses to iteratively come up with better hypotheses."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.virtual_root.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.\n",
                "- When proposing the design, take the Memory section into account.\n",
                "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.\n",
                "- Don't propose the same modelling solution but keep the evaluation the same.\n",
                "- The solution sketch should be 3-5 sentences.\n",
                "- Propose an evaluation metric that is reasonable for this task.\n",
                "- Don't suggest to do EDA.\n",
                "- The data is already prepared and available in the `./input` directory. There is no need to unzip any files.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        if "qwen3" in self.acfg.code.model:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to solve task is provided below:\n {prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"<|im_start|>system\n{introduction}<|im_end|>\n<|im_start|>user{user_prompt}<|im_end|><|im_start|>assistant\n<think>Okay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}"
        elif "deepseek" in self.acfg.code.model:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to solve task is provided below:\n{prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"<｜begin▁of▁sentence｜>\n{introduction}\n<｜User｜>{user_prompt}<｜Assistant｜><think>\nOkay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}"
        else:
            # Fallback for other models (Gemini, GPT, etc.) - use standard OpenAI chat format
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to solve task is provided below:\n {prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"{introduction}\n\n{user_prompt}\n\nPlease think step by step. First analyze the dataset:\n{self.data_preview}"
        
        self.virtual_root.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=self.virtual_root, stage="draft", local_best_node=self.virtual_root)
        logger.info(f"Drafted a new node {new_node.id} successfully!")
        return new_node

    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
        logger.info(f"Starting Improving Node {parent_node.id}.")
        introduction = (
            "As an expert AI/ML researcher, you are provided with a previously developed research implementation below "
            "and should improve it to further increase the research performance and achieve better results. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. Focus on novel, well-motivated improvements."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": parent_node.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "- The solution sketch should be a brief natural language description of how the previous solution can be improved.\n",
                "- You should be very specific and should only propose a single actionable improvement.\n",
                "- This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.\n",
                "- When proposing the design, take the Memory section into account.\n",
                "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.\n",
                "- The solution sketch should be 3-5 sentences.\n",
                "- Don't suggest to do EDA.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        output = wrap_code(parent_node.term_out, lang="")
        
        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)
        
        if "qwen3" in self.acfg.code.model:
            qwen3_user_prompt = f"\n# Task description\n{prompt['Task description']}\n# Memory\nThe memory of previous solutions used to improve performance is provided below:\n {prompt['Memory']}\n{instructions}"
            prompt_complete = f"<|im_start|>system\n{introduction}<|im_end|>\n<|im_start|>user{qwen3_user_prompt}<|im_end|><|im_start|>assistant\n<think>Okay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made attempts with the following code:\n{prompt['Previous solution']['Code']}\nThe execution of this code yielded the following results:\n{output}\nI believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
        elif "deepseek" in self.acfg.code.model:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to improve performance is provided below:\n {prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"<｜begin▁of▁sentence｜>{introduction}<｜User｜>{user_prompt}<｜Assistant｜><think>\nOkay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made attempts with the following code:\n{prompt['Previous solution']['Code']}\nThe execution of this code yielded the following results:\n{output}\nI believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
        else:
            # Fallback for other models (Gemini, GPT, etc.)
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to improve performance is provided below:\n {prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"{introduction}\n\n{user_prompt}\n\nPrevious code:\n{prompt['Previous solution']['Code']}\n\nExecution results:\n{output}\n\nPlease improve this solution. Dataset info:\n{self.data_preview}"

        parent_node.add_expected_child_count()

        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=parent_node, stage="improve", local_best_node=parent_node.local_best_node)
        logger.info(f"Improving node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
        logger.info(f"Starting Debugging Node {parent_node.id}.")
        introduction = (
            "As an expert AI/ML researcher, your previous research implementation had a bug and/or did not produce the expected results, "
            "so based on the information below, you should revise it in order to fix this. Focus on identifying and resolving the core issues. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        if self.acfg.check_format:
            introduction = (
                "As an expert AI/ML researcher, your previous research implementation had a bug and/or did not produce the expected results, or the output was in an incorrect format,"
                "so based on the information below, you should revise it in order to fix this. Focus on identifying and resolving the core issues. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "- You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.\n",
                "- Don't suggest to do EDA.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        if "qwen3" in self.acfg.code.model:
            qwen3_user_prompt = f"\n# Task description\n{prompt['Task description']}\n{instructions}"
            prompt_complete = f"<|im_start|>system\n{introduction}<|im_end|>\n<|im_start|>user{qwen3_user_prompt}<|im_end|><|im_start|>assistant\n<think>Okay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made an attempt with the following code:\n{prompt['Previous (buggy) implementation']}\nHowever, there are the following issues with this code:\n{prompt['Execution output']}\nI hold the view that the underlying reasons giving rise to the emergence of this issue are:\n{parent_node.analysis}\nThe previous solution had a bug and/or did not produce a submission.csv. I will try to fix the bug."
        elif "deepseek" in self.acfg.code.model:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n{instructions}"
            prompt_complete = f"<｜begin▁of▁sentence｜>{prompt['Introduction']}<｜User｜>{user_prompt}<｜Assistant｜><think>\nOkay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made an attempt with the following code:\n{prompt['Previous (buggy) implementation']}\nHowever, there are the following issues with this code:\n{prompt['Execution output']}\nI hold the view that the underlying reasons giving rise to the emergence of this issue are:\n{parent_node.analysis}\nThe previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format.I will try to fix the bug."
        else:
            # Fallback for other models (Gemini, GPT, etc.) - use clean format without special tokens
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n{instructions}"
            prompt_complete = f"{prompt['Introduction']}\n\n{user_prompt}\n\nPrevious buggy code:\n{prompt['Previous (buggy) implementation']}\n\nExecution output:\n{prompt['Execution output']}\n\nAnalysis:\n{parent_node.analysis}\n\nPlease fix the bugs. Dataset info:\n{self.data_preview}"
        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=parent_node, stage="debug", local_best_node=parent_node.local_best_node)
        logger.info(f"Debugging node {parent_node.id} to create new node {new_node.id}")
        return new_node
    
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = r1_query(
                prompt = prompt,
                temperature=self.acfg.code.temp,
                model=self.acfg.code.model,
                cfg=self.cfg
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore
    
    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def backpropagate(self, node: MCTSNode, value: float, add_to_tree=True):
        logger.info(f"node {node.id} start backpropagating with reward {value}.")
        while node != None:
            if node.is_buggy is False and node.parent.is_buggy is True:
                node.parent.is_debug_success = True
            elif node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
                node.parent.is_debug_success = True
            if node.parent and node.parent.stage != "root":
                node.parent.continue_improve = node.continue_improve
            if node.stage == "draft" and node.lock:
                node.lock = False
                logger.info(f"Draft node {node.id} is unlocked.")
            if node.improve_failure_depth>0:
                node.improve_failure_depth = 0
            node.update(value, add_to_tree)
            node = node.parent
            
    def parse_exec_result(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)

            introduction = (
                "As an expert AI/ML researcher, you have written code to solve this research task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings and research insights."
            )
            if self.acfg.obfuscate:
                introduction = (
                    "You are an expert machine learning engineer attempting a task. "
                    "You have written code to solve this task and now need to evaluate the output of the code execution. "
                    "You should determine if there were any bugs as well as report the empirical findings."
                )
            prompt = {
                "Introduction": introduction,
                "Task description": self.task_desc,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=review_func_spec,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    convert_system_to_user=self.acfg.convert_system_to_user,
                    cfg=self.cfg
                ),
            )

            # if the metric isn't a float then fill the metric with the worst metric
            if not isinstance(response["metric"], float):
                response["metric"] = None

            # do an extra check, to catch cases where judge fails
            has_csv_submission = (
                self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
            ).exists()

            node.analysis = response["summary"]
            if response["is_bug"] or node.exc_type is not None or response["metric"] is None or response["has_csv_submission"] == False or has_csv_submission == False:
                if response["is_bug"]:
                    logger.warning(f"Node {node.id} is marked as buggy because the response['is_bug'] is True.")
                elif node.exc_type is not None:
                    logger.warning(f"Node {node.id} is marked as buggy because the node.exc_type is not None.")
                elif response["metric"] is None:
                    logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None.")
                elif response["has_csv_submission"] == False:
                    logger.warning(f"Node {node.id} is marked as buggy because response['has_csv_submission'] is None.")
                else:
                    logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False.")

            node.is_buggy = (
                response["is_bug"]
                or node.exc_type is not None
                or response["metric"] is None
                or has_csv_submission == False
            )
            if not node.is_buggy and self.acfg.check_format:
                exp_id = self.cfg.exp_name.split("_")[0]
                logger.info(f"Start checking the format of submission.csv of node {node.id}")
                status, res = call_validate(exp_id=exp_id, submission_path=self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv")
                if status:
                    if not res['is_valid']:
                        logger.warning(f"Node {node.id} is marked as buggy because file: submission.csv is invalid.")
                        node.is_valid = False
                        node._term_out.append(f"\n{res['result']}")
                        node.analysis = "This previous solution runs without any bugs, but the format of the generated submission file is incorrect."
                    else:
                        node.is_valid = True
                        logger.info(f"Node {node.id} file: submission.csv is valid.")
                else:
                    logger.error(f"An unexpected error occurred: {res}, skip this stage.")

            if node.is_buggy:
                logger.info(
                    f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
                )
                node.metric = WorstMetricValue()
            else:
                logger.info(f"Parsed results: Node {node.id} is not buggy")
                node.metric = MetricValue(
                    response["metric"], maximize=not response["lower_is_better"]
                )
            return node
        except Exception as e:
            logger.warning(f"parse result with tool error:{e}")
            logger.info("parse_exec_result_without_tool")
            return self.parse_exec_result_without_tool(node, exec_result)

    def parse_exec_result_without_tool(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        logger.info(f"Agent is parsing execution results for node {node.id} without using tool.")
        node.absorb_exec_result(exec_result)
        introduction = (
            "As an expert AI/ML researcher, you have written code to solve this research task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings and research insights.\n\n"
            "You shoule evaluate the output of the code in Implementation. The review must be submitted in a specific JSON format with the following fields:\n\n"
            "- is_bug (boolean): This field is used to indicate whether any errors occurred during execution. If the output log shows that the execution failed or encountered a bug, set this value to true. Otherwise, set it to false.\n"
            "- has_csv_submission (boolean): This field indicates whether a submission CSV file was generated. If the code saves the predictions in a file named submission.csv in the ./submission/ directory, and it meets the required conditions, set this value to true. Otherwise, set it to false. Note that the file must be saved in the ./submission/ directory, and the filename may include a timestamp.\n"
            "- summary (string): In this field, provide a brief summary (2-3 sentences) describing the empirical findings. Alternatively, mention if there was a bug or if the submission.csv file was not properly produced. Do not suggest any fixes or improvements.\n"
            "- metric (number): If the code ran successfully, report the value of the validation metric here. If the code failed, this field should be set to null.\n"
            "- lower_is_better (boolean): This field indicates whether the metric should be minimized. If a lower value of the metric represents better performance (e.g., for Mean Squared Error), set this to true. If a higher value represents better performance (e.g., for accuracy), set this to false.\n\n"
            """The review must be submitted in the following JSON format in a single markdown code block (wrapped in ```):
```json
{
    "is_bug": true,  
    "has_csv_submission": false,  
    "summary": "The code encountered an error during execution. The CSV file was not generated.",
    "metric": null,  
    "lower_is_better": true  
}
```
"""
            ""
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        try:
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                cfg=self.cfg
            )
        except Exception as e:
            logger.info("parse without tool fail, try one more time.")
            # Convert dictionary prompt to string for r1_query
            prompt_string = f"{prompt['Introduction']}\n\nTask: {prompt['Task description']}\n\nImplementation:\n{prompt['Implementation']}\n\nExecution Output:\n{prompt['Execution output']}"
            completion_text = r1_query(
                prompt=prompt_string,
                temperature=self.acfg.code.temp,
                cfg=self.cfg
            )
        response = cast(
            dict,
            extract_review(completion_text)
        )
        if not isinstance(response["metric"], float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
        ).exists()

        node.analysis = response["summary"]
        
        if response["is_bug"] or node.exc_type is not None or response["metric"] is None or response["has_csv_submission"] == False or has_csv_submission == False:
            if response["is_bug"]:
                logger.warning(f"Node {node.id} is marked as buggy because the response['is_bug'] is True.")
            elif node.exc_type is not None:
                logger.warning(f"Node {node.id} is marked as buggy because the node.exc_type is not None.")
            elif response["metric"] is None:
                logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None.")
            elif response["has_csv_submission"] == False:
                logger.warning(f"Node {node.id} is marked as buggy because response['has_csv_submission'] is None.")
            else:
                logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False.")

        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
            or has_csv_submission == False
        )
        if not node.is_buggy and self.acfg.check_format:
            exp_id = self.cfg.exp_name.split("_")[0]
            logger.info(f"Start checking the format of submission.csv of node {node.id}")
            status, res = call_validate(exp_id=exp_id, submission_path=self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv")
            if status:
                if not res['is_valid']:
                    logger.warning(f"Node {node.id} is marked as buggy because file: submission.csv is invalid.")
                    node.is_valid = False
                    node._term_out.append(f"\n{res['result']}")
                    node.analysis = "This previous solution runs without any bugs, but the format of the generated submission file is incorrect."
                else:
                    node.is_valid = True
                    logger.info(f"Node {node.id} file: submission.csv is valid.")
            else:
                logger.error(f"An unexpected error occurred: {res}, skip this stage.")

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
        return node

    def select(self, node: MCTSNode):
        logger.info(f"[select] Processing node: {node.id}")
        while node and not node.is_terminal:
            if not node.is_fully_expanded_with_expected(scfg=self.scfg):
                if node.is_buggy and node.is_debug_success is True:
                    node = self.uct_select(node)
                elif node.continue_improve and len(node.children)>0:
                    node = self.uct_select(node)
                else:
                    logger.info(f"Node {node.id} is not fully expanded, expanding")
                    return node
            else:
                node = self.uct_select(node)
        logger.info(f"[select]choose a node for expanding: {node.id}")
        return node

    def get_C(self):
        dcfg =  self.cfg.agent.decay
        if dcfg.decay_type == "linear":
            linear_cfg = dcfg.linear_decay
            return linear_decay(
                t=self.current_step, 
                initial_C=dcfg.exploration_constant,
                lower_bound=dcfg.lower_bound,
                alpha=linear_cfg.alpha
            )
        
        elif dcfg.decay_type == "exponential":
            exponential_cfg = dcfg.exponential_decay
            return exponential_decay(
                t=self.current_step,
                initial_C=self.scfg.exploration_constant,
                lower_bound=dcfg.lower_bound,
                gamma=exponential_cfg.gamma,
            )
        
        elif dcfg.decay_type == "piecewise":
            piecewise_cfg = dcfg.piecewise_decay
            n1 = self.scfg.num_drafts*(self.scfg.num_improves ** 2)
            n2 = round(self.acfg.steps*piecewise_cfg.phase_ratios[0])
            t1 = min(n1,n2)
            t2 = round(self.acfg.steps*piecewise_cfg.phase_ratios[1])
            return piecewise_decay(
                t=self.current_step, 
                initial_C=dcfg.exploration_constant,
                T1=t1,
                T2=t2,
                lower_bound=dcfg.lower_bound
            )
        
        elif dcfg.decay_type == "dynamic_piecewise":
            dynamic_piecewise_cfg = dcfg.dynamic_piecewise_decay
            logger.info(f"dynamic_piecewise_cfg.phase_ratios = {dynamic_piecewise_cfg.phase_ratios}")
            return dynamic_piecewise_decay(
                steps_limit=self.acfg.steps,
                n_nodes=self.current_step,
                initial_C=dcfg.exploration_constant,
                start_time=self.search_start_time,
                time_limit=self.acfg.time_limit,
                alpha=dynamic_piecewise_cfg.alpha,
                lower_bound=dcfg.lower_bound,
                phase_ratios=dynamic_piecewise_cfg.phase_ratios
            )
        else:
            return dcfg.exploration_constant

    def uct_select(self, node: MCTSNode):
        if self.is_root(node):
            filtered_children = [child for child in node.children if not child.lock]
            logger.info(f"For node {node.id}, there are {len(node.children) - len(filtered_children)}/{len(node.children)} is locked.")
            selected_node = node
            if len(filtered_children) > 0:
                selected_node = max(filtered_children, key=lambda child: child.uct_value(exploration_constant = self.get_C()))
                
            if selected_node.stage == "draft":
                selected_node.lock = True
                logger.info(f"Draft node {selected_node.id} is locked.")
            return selected_node
        else:
            return max(node.children, key=lambda child: child.uct_value(exploration_constant = self.get_C()))

    
    def check_improvement(self, cur_node: MCTSNode, parent_node: MCTSNode):
        improvement = 0
        should_backpropagate = False
        local_best_node = cur_node.local_best_node
        local_best_metric = local_best_node.metric.value
        if cur_node.is_buggy is False:
            new_metric = cur_node.metric.value  
            if parent_node.is_buggy:
                logger.info(f"Successfully Debug the error in node {parent_node.id}.")
                should_backpropagate = True
            if new_metric and local_best_metric:
                improvement = new_metric - local_best_metric if cur_node.metric.maximize else local_best_metric - new_metric
                if improvement < self.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < self.scfg.max_improve_failure:
                    local_best_node.improve_failure_depth += 1
                    logger.warning(f"Compared to Node {local_best_node.id}, Node {cur_node.id} metric improvement ({improvement}) below threshold ({self.scfg.metric_improvement_threshold}), try one more time({local_best_node.improve_failure_depth}/{self.scfg.max_improve_failure})")
                    cur_node.continue_improve = True
                elif improvement < self.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= self.scfg.max_improve_failure:
                    logging.warning(f"The number of improvement attempts for the local best node has reached its maximum limit {self.scfg.max_improve_failure}.")
                    cur_node.continue_improve = False
                    should_backpropagate = True
                    cur_node.is_terminal = True
                else:
                    logger.info(f"Compared to Node {local_best_node.id}, Node {cur_node.id} metric improvement ({improvement}) above threshold ({self.scfg.metric_improvement_threshold}), continue improving.")
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
            elif new_metric:
                logger.info(f"No local best node was found among the previous nodes; the current node {cur_node.id} is assigned as the local best")
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            else:
                logger.warning(f"No local best node was found among the previous nodes; The current node {cur_node.id} has no errors, but contains an empty metric value.")
                should_backpropagate = True
        elif cur_node.is_buggy is None:
            logger.warning(f"Node {cur_node.id} is_buggy is None!")
            should_backpropagate = True
        else:
            if cur_node.debug_depth >= self.scfg.back_debug_depth:
                should_backpropagate = True
                if cur_node.debug_depth >= self.scfg.max_debug_depth:
                    cur_node.is_terminal = True

        if should_backpropagate:
            reward = self.get_node_reward(cur_node)
            self.backpropagate(cur_node, reward)
        else:
            self.current_node_list.append(cur_node)
        return should_backpropagate
    
    def get_node_reward(self, node: MCTSNode):
        reward = 0
        if node.is_buggy is True or node.is_buggy is None:
            reward = -1
        elif node.is_buggy is False and node.metric.value is None:
            reward = -1
        else:
            if node.metric.value and self.best_metric:
                improvement = node.metric.value - self.best_metric if node.metric.maximize else self.best_metric - node.metric.value
                if improvement > 0:
                    logger.info(f"Node {node.id} is better than the best node {self.best_node.id} now!")
                    reward += 1
            if node.parent.is_buggy is True:
                reward += 1
            else:
                reward += 1
        return reward
            
    def is_root(self, node: MCTSNode):
        return node.id is self.virtual_root.id
    
    def check_metric_valid(self, node: MCTSNode, upper_bound=50):
        '''If the metric values between nodes differ by an upper bound multiple, it is highly likely that there is an invalid metric'''
        upper_bound = self.acfg.search.invalid_metric_upper_bound if self.acfg.search.invalid_metric_upper_bound else upper_bound
        v1 = self.best_metric
        v2 = node.metric.value
        if v1 is None or v2 is None:
            return True
        elif v1 == 0 or v2 == 0:
            return abs(v1 - v2) <= upper_bound
        else:
            ratio = max(abs(v1), abs(v2)) / min(abs(v1), abs(v2))
            return ratio <= upper_bound

    def _step_search(self, parent_node: MCTSNode, exec_callback: ExecCallbackType):
        logger.info(f"[_step_search] Processing node: {parent_node.id}")
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")
        result_node = None
        _root = False
    
        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    result_node = self._draft()
                    result_node.lock = True
                    logger.info(f"[_step_search]Draft node {result_node.id} is locked.")
                elif parent_node.is_buggy or parent_node.is_valid is False:
                    result_node = self._debug(parent_node)
                elif parent_node.is_buggy is False:
                    result_node = self._improve(parent_node)
                else:
                    logger.warning(f"[_step_search] node {parent_node.id} is_buggy is None.")
                
                if result_node:
                
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                
                    result_node = self.parse_exec_result(
                        node=result_node,
                        exec_result=exe_res
                    )
                    if not result_node.is_buggy:
                        if not (self.cfg.workspace_dir / "submission" / f"submission_{result_node.id}.csv").exists():
                            result_node.is_buggy = True
                            result_node.metric = WorstMetricValue()
                            logger.info(f"Actually, node {result_node.id} did not produce a submission.csv")
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    if not self.check_metric_valid(node=result_node):
                        result_node.metric = WorstMetricValue()
                        logger.info(f"node {result_node.id} generate invalid metric.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True
                    
                    _root = self.check_improvement(result_node, parent_node)
                    with self.journal_lock:
                        if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning("New node's metric is inconsistent with metrics in the journal.Returning to the parent node to regenerate.")
                            raise ValueError("New node's metric is inconsistent with metrics in the journal.Returning to the parent node to regenerate.")
                        else:
                            self.journal.append(result_node)
                            

            except Exception as e:
                logger.warning("Current node generation failed, rolling back to unlock the draft node.")
                self.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
                raise e

        else:
            logger.info(f"current node is terminal, backpropagating!!")
            self.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node
    
    def get_best_node(self, node_list):
        good_node = [n for n in node_list if not n.is_buggy and n.metric]
        if not good_node:
            return None
        return max(good_node, key=lambda n: n.metric)

    def step(self, node: MCTSNode, exec_callback: ExecCallbackType) -> bool:   
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        if not node or node.stage == "root":
            node = self.select(self.virtual_root)

        _root, result_node = self._step_search(node, exec_callback=exec_callback)
        if result_node:
            submission_file_path = self.cfg.workspace_dir / "submission" / f"submission_{result_node.id}.csv"
            logger.info(f"In the search step from node {node.id}, the generated node is {result_node.id}, the metric is {result_node.metric.value}")
        if result_node and result_node.metric.value is not None:
            if self.best_node is None or self.best_node.metric < result_node.metric:
                logger.info(f"Node {result_node.id} is the best node so far")
                if self.best_node is None or result_node.is_valid is True:
                    self.best_node = result_node
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    with self.save_node_lock:
                        best_solution_dir.mkdir(exist_ok=True, parents=True)
                        best_submission_dir.mkdir(exist_ok=True, parents=True)
                        shutil.copy(
                            submission_file_path,
                            best_submission_dir / "submission.csv",
                        )
                        with open(best_solution_dir / "solution.py", "w") as f:
                            f.write(result_node.code)
                        with open(best_solution_dir / "node_id.txt", "w") as f:
                            f.write(str(result_node.id))
                else:
                    logger.info(f"Node {result_node.id} is a invalid node")
                    logger.info(f"Node {self.best_node.id} is still the best node")
            else:
                if self.best_node.is_valid is False:
                    logger.info(f"Node {self.best_node.id} is invalid, {result_node.id} is the best node so far")
                    self.best_node = result_node
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    with self.save_node_lock:
                        best_solution_dir.mkdir(exist_ok=True, parents=True)
                        best_submission_dir.mkdir(exist_ok=True, parents=True)
                        shutil.copy(
                            submission_file_path,
                            best_submission_dir / "submission.csv",
                        )
                        with open(best_solution_dir / "solution.py", "w") as f:
                            f.write(result_node.code)
                        with open(best_solution_dir / "node_id.txt", "w") as f:
                            f.write(str(result_node.id))

                else:
                    logger.info(f"Node {result_node.id} is not the best node")
                    logger.info(f"Node {self.best_node.id} is still the best node")
        elif not result_node:
            logger.info(f"Result node is None.")
        else:
            logger.info(f"result node has bug.")
        if self.best_node:
            logger.info(f"Best metric value is {self.best_node.metric.value}.")

        if not self.acfg.save_all_submission and result_node and os.path.exists(submission_file_path):
            os.remove(submission_file_path)
        self.current_step = len(self.journal)
        if _root or result_node is None:
            logger.info(f"agent return root to main")
            return self.virtual_root
        else:
            logger.info(f"agent return {result_node.id} to main")
            return result_node
        