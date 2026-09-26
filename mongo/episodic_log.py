"""
Episodic log: append-only record of every tool call/result, written to
MongoDB instead of only living in the live prompt.

nyuctf_multiagent/conversation.py already truncates what the model sees
(truncate_content, len_observations) to protect the context window -- this
module doesn't change that. It just means nothing is actually lost when the
conversation truncates it: the full, untruncated result is preserved here,
out of the LLM's context but still queryable.

Hook point: BaseAgent.add_observation_message() in nyuctf_multiagent/agent.py
calls the agent's `observation_logger` *before* Conversation.append_observation()
truncates the result in place. RAGPlannerExecutorSystem attaches an
EpisodicLogger to every agent in mongo_rag mode.
"""
import json
import time
from pymongo.collection import Collection

# A MongoDB document is capped at 16MB; keep each logged output well under it
MAX_LOGGED_CHARS = 1_000_000


def _as_text(result) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


class EpisodicLogger:
    """Writes one `episodic_log` document per tool result of one episode."""

    def __init__(self, collection: Collection, episode_id: str):
        self.collection = collection
        self.episode_id = episode_id
        self.count = 0
        self.failed = False
        try:
            self.collection.create_index([("episode_id", 1), ("seq", 1)])
        except Exception:
            pass  # index is an optimisation only

    def log(self, agent_role: str, round_num: int, tool_name: str, tool_call_id,
            result, prompt_limit_chars: int = None):
        full_output = _as_text(result)
        doc = {
            "episode_id": self.episode_id,
            "seq": self.count,
            "round": round_num,
            "agent_role": agent_role,
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "full_output": full_output[:MAX_LOGGED_CHARS],
            "output_chars": len(full_output),
            # True when the LLM only saw a cut-down version of this output
            "truncated_in_prompt": prompt_limit_chars is not None and len(full_output) > prompt_limit_chars,
            "stored_truncated": len(full_output) > MAX_LOGGED_CHARS,
            "timestamp": time.time(),
        }
        try:
            self.collection.insert_one(doc)
            self.count += 1
        except Exception as e:
            # Logging must never take the agent down; report once and carry on
            if not self.failed:
                self.failed = True
                print(f"[Episodic log] write to MongoDB failed, continuing without it: {e}")


def log_tool_result(collection: Collection, episode_id: str, round_num: int,
                     tool_name: str, full_output, agent_role: str = "executor"):
    """One-off helper kept for scripts; the agent uses EpisodicLogger."""
    collection.insert_one({
        "episode_id": episode_id,
        "round": round_num,
        "agent_role": agent_role,
        "tool": tool_name,
        "full_output": _as_text(full_output),
        "timestamp": time.time(),
    })
