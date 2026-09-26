"""
Call this once at the end of every episode -- right after
self.environment.solved is known in rag_agent.py's run(). It:

  1. Updates utility_score on every memory that was actually injected (outcome tracking).
  2. Logs a `runs` document for the Atlas Charts dashboard (live proof).
  3. Re-evaluates the self-tuning policy (self-tuning rules).

This is the glue for pieces 2, 5, and 6 of the plan. It can also be run
standalone from the command line to test the pipeline against fake data
before wiring it into the real agent loop -- see the __main__ block.
"""
import sys
import time
from pathlib import Path

from pymongo.collection import Collection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mongo.mongo_rag import record_outcome
from mongo.policy import update_policy


def finish_episode(memory_col: Collection, runs_col: Collection, policy_col: Collection,
                    episode_id: str, category: str, challenge_name: str,
                    injected_doc_ids, solved: bool, tokens_used: int, cost: float,
                    policy_version: int):
    record_outcome(memory_col, injected_doc_ids, solved)

    runs_col.insert_one({
        "episode_id": episode_id,
        "category": category,
        "challenge": challenge_name,
        "solved": solved,
        "tokens_used": tokens_used,
        "cost": cost,
        "policy_version": policy_version,
        "timestamp": time.time(),
    })

    return update_policy(policy_col, runs_col)


if __name__ == "__main__":
    import argparse

    from pymongo import MongoClient
    from nyuctf_multiagent.utils import APIKeys

    parser = argparse.ArgumentParser(description="Manually log one episode's outcome (for testing the pipeline).")
    parser.add_argument("--keys", default="keys.cfg")
    parser.add_argument("--db", default="pentest_memory")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--doc-ids", nargs="*", default=[], help="doc_id values that were injected this episode")
    parser.add_argument("--solved", action="store_true")
    parser.add_argument("--tokens-used", type=int, required=True)
    parser.add_argument("--cost", type=float, default=0.0)
    args = parser.parse_args()

    keys = APIKeys(args.keys)
    mongo = MongoClient(keys["MONGODB_URI"])
    db = mongo[args.db]

    changes = finish_episode(
        memory_col=db["semantic_memory"], runs_col=db["runs"], policy_col=db["policy"],
        episode_id=args.episode_id, category=args.category, challenge_name=args.challenge,
        injected_doc_ids=args.doc_ids, solved=args.solved, tokens_used=args.tokens_used,
        cost=args.cost, policy_version=1,
    )
    print(f"Logged episode {args.episode_id}. Policy change: {changes or 'none this time'}")
