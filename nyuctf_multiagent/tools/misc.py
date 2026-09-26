from ..logging import logger

from .tool import Tool

class SubmitFlagTool(Tool):
    NAME = "submit_flag"
    DESCRIPTION = "Submit the flag for the CTF challenge. If the flag is correct, the challenge ends with success, else this command returns error and you can continue with the challenge."
    PARAMETERS = {
        "flag": ("string", "the flag"),
    }
    REQUIRED_PARAMETERS = {"flag"}

    def __init__(self, environment):
        super().__init__()
        self.environment = environment
        self.actual_flag = environment.challenge.flag

    def call(self, flag=None):
        if flag is None:
            return {"error": "Flag not provided!"}
        if flag != self.actual_flag:
            return {"error": "Flag is wrong!"}
        self.environment.solved = True
        return {"success": True}

    def print_tool_call(self, tool_call):
        logger.assistant_action(f"**{self.NAME}:** `{tool_call.parsed_arguments['flag']}`")

    def print_result(self, tool_result):
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            logger.print(f"[green bold]SOLVED![/green bold]", markup=True)

class GiveupTool(Tool):
    NAME = "giveup"
    DESCRIPTION = "Give up this challenge to terminate it immediately. Use this to stop solving the challenge."
    PARAMETERS = {
        "confirm": ("boolean", "confirm give up"),
    }
    REQUIRED_PARAMETERS = set()

    def __init__(self, environment):
        super().__init__()
        self.environment = environment

    def call(self, confirm=True):
        self.environment.giveup = True
        return {"success": True}

    def print_tool_call(self, tool_call):
        logger.assistant_action(f"**{self.NAME}**")

    def print_result(self, tool_result):
        logger.print(f"[red bold]AGENT GAVE UP![/red bold]", markup=True)

class DelegateTool(Tool):
    """Tool to delegate task from planner to executor"""
    NAME = "delegate"
    DESCRIPTION = "Delegate a task to an executor LLM agent. You have TWO executors: 'executor_a' is a general-purpose executor for reconnaissance, scripting, web exploitation, forensics, and cryptography tasks. 'executor_b' is a binary-analysis specialist with access to Ghidra disassembly/decompilation tools, ideal for reverse engineering and pwn/binary exploitation tasks. Choose the appropriate executor for the task."
    PARAMETERS = {
        "task": ("string", "a detailed task description."),
        "executor_type": ("string", "which executor to delegate to: 'executor_a' (general) or 'executor_b' (binary/reversing). Default is 'executor_a'."),
    }
    REQUIRED_PARAMETERS = {"task"}
    def __init__(self, environment):
        super().__init__()

    def print_tool_call(self, tool_call):
        executor_type = tool_call.parsed_arguments.get('executor_type', 'executor_a')
        logger.assistant_action(f"**{self.NAME}** → [{executor_type}]\n\n{tool_call.parsed_arguments['task']}")
    def print_result(self, tool_result):
        # TODO move the error check to common
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            # TODO no need to print anything?
            pass

class GenAutoPromptTool(Tool):
    """Tool to autonomously generate prompt and return prompt and summary to the Planner."""
    NAME = "generate_prompt"
    DESCRIPTION = "Generate a prompt for the Planner agent to solve the CTF challenge."
    PARAMETERS = {
        "prompt": ("string", "the prompt for the Planner agent."),
    }
    REQUIRED_PARAMETERS = {"prompt"}
    def __init__(self, environment):
        super().__init__()

    def print_tool_call(self, tool_call):
        logger.assistant_action(f"**{self.NAME}**\n\n{tool_call.parsed_arguments['prompt']}")
    def print_result(self, tool_result):
        # TODO move the error check to common
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            # TODO no need to print anything?
            pass

class FinishTaskTool(Tool):
    """Tool to mark executor task finished and return summary to planner."""
    NAME = "finish_task"
    DESCRIPTION = "Finish the task assigned by the planner and return the task summary."
    PARAMETERS = {
        "summary": ("string", "a detailed summary of the task performed"),
    }
    REQUIRED_PARAMETERS = {"summary"}
    def __init__(self, environment):
        super().__init__()

    def print_tool_call(self, tool_call):
        logger.assistant_action(f"**{self.NAME}**\n\n{tool_call.parsed_arguments['summary']}")
    def print_result(self, tool_result):
        # TODO move the error check to common
        if "error" in tool_result.result:
            logger.print(f"[bold]{self.NAME}[/bold]: [red]{tool_result.result['error']}[/red]", markup=True)
        else:
            # TODO no need to print anything?
            pass

