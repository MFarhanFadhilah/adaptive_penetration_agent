"""
Episodic log for Mongo-RAG.

Stores the full, untruncated result of every tool call in an episode
(planner, autoprompter, and every executor) in the `episodic_log` collection,
keyed by `episode_id`. The in-conversation window only ever sees a truncated
version of each result (see Conversation.truncate_content); this is the raw
material that outcome tracking and the demo webapp read back from Atlas.
"""
import time


class EpisodicLogger:
    def __init__(self, collection, episode_id: str):
        self.collection = collection
        self.episode_id = episode_id
        self.count = 0

    def log(self, agent_role, round_num, tool_name, tool_call_id, result, prompt_limit_chars):
        self.collection.insert_one({
            "episode_id": self.episode_id,
            "agent_role": agent_role,
            "round": round_num,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "result": result,
            "prompt_limit_chars": prompt_limit_chars,
            "timestamp": time.time(),
        })
        self.count += 1
