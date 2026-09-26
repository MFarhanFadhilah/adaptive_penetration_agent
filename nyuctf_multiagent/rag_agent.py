"""
RAG-Augmented Planner-Executor System.

Extends PlannerExecutorSystem (from agent.py) with Self-RAG and/or Graph-RAG:
  - Before the planner starts each round, RAG modules are queried.
  - Retrieved hints are injected into the planner's context as a user message.
  - All injection events are recorded in the JSON log output.

Usage:
  system = RAGPlannerExecutorSystem(
      ...,
      self_rag=SelfRAG(...),
      graph_rag=GraphRAG(...),
  )
  with system:
      system.run()
"""
import json
import time
import uuid
from pathlib import Path

from .agent import PlannerExecutorSystem, PlannerAgent, ExecutorAgent, AutoPromptAgent
from .logging import logger
from .tools import DelegateTool, ToolResult


class RAGPlannerExecutorSystem(PlannerExecutorSystem):
    """
    Planner-Executor system augmented with Self-RAG and/or Graph-RAG.

    At the beginning of the run (round 1) and every `rag_inject_every` rounds,
    queries both RAG modules, merges their hints, and injects them as a
    formatted user message into the planner conversation before the LLM call.
    """

    def __init__(
        self,
        environment,
        challenge,
        autoprompter,
        planner,
        executor,
        max_cost: float = 1.0,
        logfile=None,
        executor_b=None,
        self_rag=None,
        graph_rag=None,
        mongo_rag=None,
        runs_collection=None,
        policy_collection=None,
        policy_version=None,
        episodic_collection=None,
        rag_inject_every: int = 3,
        mode: str = "self_rag",   # "self_rag", "graph_rag", "mongo_rag", or "combined"
    ):
        super().__init__(
            environment, challenge, autoprompter, planner, executor,
            max_cost=max_cost, logfile=logfile, executor_b=executor_b
        )
        self.self_rag = self_rag
        self.graph_rag = graph_rag
        self.mongo_rag = mongo_rag
        self.runs_collection = runs_collection
        self.policy_collection = policy_collection
        self.policy_version = policy_version
        # One id ties this episode's `runs` document to its `episodic_log` documents
        self.episode_id = f"{challenge.name}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.episodic_logger = None
        if episodic_collection is not None:
            from mongo.episodic_log import EpisodicLogger
            self.episodic_logger = EpisodicLogger(episodic_collection, self.episode_id)
        self.rag_inject_every = rag_inject_every
        self.mode = mode
        self._rag_injections: list = []

    # ------------------------------------------------------------------
    # RAG injection helpers
    # ------------------------------------------------------------------

    def _build_rag_context(self, current_round: int) -> str:
        """Query RAG modules and build a combined hint block."""
        hints_text_parts = []
        challenge_name = self.challenge.name
        challenge_category = self.challenge.category
        challenge_description = self.challenge.description or ""

        # Summarise current planner conversation as plan context
        recent_messages = self.planner.conversation.all_messages[-4:]
        current_plan = " ".join(
            m.content for m in recent_messages if m.content is not None
        )

        if self.self_rag and self.mode in ("self_rag", "combined"):
            hints = self.self_rag.get_hints(
                challenge_name=challenge_name,
                challenge_category=challenge_category,
                challenge_description=challenge_description,
                current_plan=current_plan,
                current_round=current_round,
            )
            block = self.self_rag.format_hints_for_injection(hints)
            if block:
                hints_text_parts.append(block)
                self._rag_injections.append({
                    "type": "self_rag",
                    "round": current_round,
                    "num_hints": len(hints),
                    "hints": [h.to_dict() for h in hints],
                })
                logger.print(
                    f"[Self-RAG] Injecting {len(hints)} hint(s) at round {current_round}",
                    style="cyan bold",
                    force=True,
                )

        if self.graph_rag and self.mode in ("graph_rag", "combined"):
            hints = self.graph_rag.get_hints(
                challenge_name=challenge_name,
                challenge_category=challenge_category,
                challenge_description=challenge_description,
                current_plan=current_plan,
                current_round=current_round,
            )
            block = self.graph_rag.format_hints_for_injection(hints)
            if block:
                hints_text_parts.append(block)
                self._rag_injections.append({
                    "type": "graph_rag",
                    "round": current_round,
                    "num_hints": len(hints),
                    "hints": [h.to_dict() for h in hints],
                })
                logger.print(
                    f"[Graph-RAG] Injecting {len(hints)} graph hint(s) at round {current_round}",
                    style="magenta bold",
                    force=True,
                )

        if self.mongo_rag and self.mode in ("mongo_rag", "combined"):
            hints = self.mongo_rag.get_hints(
                challenge_name=challenge_name,
                challenge_category=challenge_category,
                challenge_description=challenge_description,
                current_plan=current_plan,
                current_round=current_round,
            )
            block = self.mongo_rag.format_hints_for_injection(hints)
            if block:
                hints_text_parts.append(block)
                self._rag_injections.append({
                    "type": "mongo_rag",
                    "round": current_round,
                    "num_hints": len(hints),
                    "hints": [h.to_dict() for h in hints],
                })
                logger.print(
                    f"[Mongo-RAG] Injecting {len(hints)} hint(s) at round {current_round}",
                    style="green bold",
                    force=True,
                )

        return "\n\n".join(hints_text_parts)

    def _inject_rag_hints(self, round_num: int):
        """Inject RAG hints into planner conversation if this is an injection round."""
        if round_num == 1 or round_num % self.rag_inject_every == 0:
            hint_block = self._build_rag_context(round_num)
            if hint_block:
                injection_msg = (
                    f"[RAG Knowledge Injection - Round {round_num}]\n\n"
                    f"{hint_block}\n\n"
                    "Use the above hints to refine your plan if they are relevant. "
                    "Ignore any hints that do not apply to this challenge. "
                    "Continue solving the challenge."
                )
                self.planner.add_user_message(injection_msg)

    # ------------------------------------------------------------------
    # Override run() to add RAG injection
    # ------------------------------------------------------------------

    def _log_observation(self, agent, tool_result):
        """observation_logger callback: store the full, untruncated tool result."""
        if agent is self.planner:
            role = "planner"
        elif agent is self.autoprompter:
            role = "autoprompter"
        else:
            idx = next((i for i, e in enumerate(self.all_executors) if e is agent), None)
            role = "executor" if idx is None else f"executor_{idx}"
        self.episodic_logger.log(
            agent_role=role,
            round_num=agent.conversation.round,
            tool_name=tool_result.name,
            tool_call_id=tool_result.id,
            result=tool_result.result,
            prompt_limit_chars=agent.conversation.truncate_content,
        )

    def run(self):
        if self.episodic_logger is not None:
            # Executors are cloned per task via ExecutorAgent.new(), which carries this over
            for agent in (self.autoprompter, self.planner, self.executor, self.executor_b):
                if agent is not None:
                    agent.observation_logger = self._log_observation

        planner_initial = self.planner.prompter.get("initial")

        if self.autoprompter.enabled:
            self.run_autoprompter()
            if self.autoprompter.autoprompt is not None:
                planner_initial = self.autoprompter.autoprompt
            elif not self.environment.solved:
                logger.print(
                    "WARNING! Autoprompter failed to generate a prompt, using hardcoded one",
                    force=True,
                    style="dark_orange bold",
                )

        logger.print("============= RAG-PLANNER ===============", style="bold")
        self.planner.add_system_message(self.planner.prompter.get("system"))
        self.planner.add_user_message(planner_initial)

        # Inject RAG at round 1 (before planner's first LLM call)
        self._inject_rag_hints(1)

        planner_round = 0
        while (
            not self.environment.solved
            and not self.environment.giveup
            and self.planner.conversation.round <= self.planner.max_rounds
            and self.total_cost() <= self.max_cost
        ):
            planner_round += 1
            self.planner.conversation.next_round()
            self.planner.run_one_round()

            if self.planner.delegated_task is not None:
                executor_type = "executor_a"
                if self.planner.delegated_task.parsed_arguments is not None:
                    executor_type = self.planner.delegated_task.parsed_arguments.get(
                        "executor_type", "executor_a"
                    )
                result = self.run_executor(self.planner.delegated_task, executor_type=executor_type)
                tool_result = ToolResult(
                    name=DelegateTool.NAME,
                    id=self.planner.delegated_task.id,
                    result=result,
                )
                self.planner.add_observation_message(tool_result)
                self.planner.delegated_task = None

                # Inject RAG hints after executor finishes (every Nth round)
                if planner_round % self.rag_inject_every == 0:
                    self._inject_rag_hints(planner_round)

        # Report this episode's outcome back to MongoDB. This single insert is
        # what the Atlas Trigger (outcome tracking) and Atlas Charts (live
        # proof) both key off -- see MONGODB_ATLAS_UI_ONLY.md Step 7.
        if self.mongo_rag is not None and self.runs_collection is not None:
            input_tokens, output_tokens = self.total_tokens()
            tokens_used = input_tokens + output_tokens
            self.runs_collection.insert_one({
                "episode_id": self.episode_id,
                "category": self.challenge.category,
                "challenge": self.challenge.name,
                "solved": self.environment.solved,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": self.total_cost(),
                "used_doc_ids": self.mongo_rag.episode_injections,
                "policy_version": self.policy_version,
                "tool_calls_logged": self.episodic_logger.count if self.episodic_logger else 0,
                "timestamp": time.time(),
            })
            self._update_policy(tokens_used)

    def _update_policy(self, tokens_used: int):
        """Let the self-tuning policy react to this run (same rule the demo webapp uses)."""
        if self.policy_collection is None:
            return
        if tokens_used <= 0:
            # A 0 would drag the average down and wrongly relax the policy, so
            # skip tuning until real token counts are recorded.
            logger.print("[Policy] Skipping self-tuning: this run has no token count yet",
                         style="yellow", force=True)
            return
        from mongo.policy import update_policy
        changes = update_policy(self.policy_collection, self.runs_collection)
        if changes:
            logger.print(f"[Policy] Updated in Atlas: {changes}", style="green bold", force=True)
        else:
            logger.print("[Policy] No change this run", force=True)

    # ------------------------------------------------------------------
    # Override dump_log to include RAG trace
    # ------------------------------------------------------------------

    def dump_log(self, error=None):
        # First write the standard log fields via the base class internal method
        if self.logfile is None:
            return

        exit_reason = "error" if error is not None else self.get_exit_reason()
        cost = self.total_cost()
        with self.logfile.open("w") as lf:
            json.dump(
                {
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "time_taken": (self.end_time - self.start_time),
                    "rag_mode": self.mode,
                    "autoprompter_model": None
                    if not self.autoprompter.enabled
                    else self.autoprompter.backend.model,
                    "planner_model": self.planner.backend.model,
                    "executor_model": self.executor.backend.model,
                    "executor_b_model": self.executor_b.backend.model
                    if self.executor_b is not None
                    else None,
                    "total_cost": cost,
                    "total_tokens": sum(self.total_tokens()),
                    "success": self.environment.solved,
                    "exit_reason": exit_reason,
                    "error": error,
                    # RAG-specific sections
                    "rag_injections": self._rag_injections,
                    "self_rag_log": self.self_rag.dump_log() if self.self_rag else [],
                    "graph_rag_log": self.graph_rag.dump_log() if self.graph_rag else [],
                    "mongo_rag_log": self.mongo_rag.dump_log() if self.mongo_rag else [],
                    # Conversation traces
                    "autoprompter": []
                    if not self.autoprompter.enabled
                    else self.autoprompter.conversation.dump(),
                    "planner": self.planner.conversation.dump(),
                    "executors": [e.conversation.dump() for e in self.all_executors],
                    "executor_errors": [e.error for e in self.all_executors],
                    "debug_log": logger.debug_log,
                },
                lf,
                indent=2,
            )

        if exit_reason == "solved":
            logger.print(
                "[green bold]Challenge Solved![/green bold]", force=True, markup=True
            )
        else:
            logger.print(
                "[red bold]Challenge Not Solved![/red bold]", force=True, markup=True
            )
        logger.print(
            f"exit: {exit_reason} cost: ${cost:.3f} "
            f"planner-rounds: {self.planner.conversation.round} "
            f"num-executors: {len(self.all_executors)} "
            f"rag-injections: {len(self._rag_injections)}",
            force=True,
        )
