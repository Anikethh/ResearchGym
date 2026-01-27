from string import Template

additional_notes_template = Template(
    """
ADDITIONAL NOTES
------
${idea_hint_note}- **Compute**: You have access to a ${type_of_processor} with the appropriate drivers installed.
- **Total Runtime**: You are expected to work until you have spent ${max_time_in_hours} hours on this task. You may iteratively develop your solution within these constraints, but bear in mind that program runtime also counts towards the time limit. This is a difficult task and we expect a good solution to require using the full time limit.
- **API keys**: You have been granted API keys for services necessary for the task. The API keys are available in the `.env` file at the root of your workspace. You can assume that these same API keys will already be set as environment variables in the reproduction environment when we evaluate your submission, so you do not need to set or export them in your submission.
- You have root access in your environment.
- You can use the web search tool to search for finding relevant information on the web such as research papers, blogs, software documentations etc. You can also use the semantic scholar APIs to search for papers. You are encouraged to seek inspirations and learn about relevant research on the concerned topic.
- ${literature_line}${hypothesis_line}The results should beat the baselines with a significant margin. Do not use fallbacks, dummy datasets or hardcode any componenet during the development process.
- You can use git for version control, avoid tracking large artefacts like datasets and model checkpoints. You can maintain a .gitignore for such formats, if required. 
- Be mindful of running grep commands that scan through all files in the directory as it might take too long to execute.
- ${multiple_hypotheses_line}You must use all the time available to you. Attempt to have multiple results for your proposed methods before the time runs out.
- You would have multiple sub-tasks that can you should achieve results for, ensure you have some results on the primary sub-task before attempting other sub-tasks.
- You are also encouraged to note down your ideas, overall progress, analysis, limitations, etc. to keep track of your previous actions, as this is a long running research task, you should ensure to take notes of important steps that you might need for reference later.
- Only use the end task tool if you have exhausted all your ideas and you are confident you cannot further improve the performance on the benchmark.
- Remember, you must actually write code, run experiments to improve scores over the baselines, not just write a plan for how to do so.
${async_note}
"""
)
