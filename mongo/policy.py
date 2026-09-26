"""
Self-tuning policy document: replaces the static YAML in
configs/rag/self_rag_config.yaml with a live MongoDB document the harness
reads at startup and can rewrite based on measured outcomes -- not code,
not a config file someone has to edit by hand.

There are two independent policy documents in the `policy` collection, so the
demo webapp can never retune the real agent:

  _id "global" -- the real CTF agent (run_rag.py --rag-mode mongo_rag);
                  tuned only from real runs in `runs`
  _id "demo"   -- the demo webapp; tuned only from runs marked simulated=True

Each document carries its own `token_budget`, editable live in Atlas.
"""
import time
from pymongo.collection import Collection

POLICY_REAL = "global"
POLICY_DEMO = "demo"

DEFAULT_POLICY = {
    "version": 1,
    "rag_top_k": 3,
    "rag_inject_every": 3,
    "truncate_content_chars": 25000,
    "alpha": 0.6,   # weight on semantic similarity in MongoRAG ranking
    "beta": 0.4,    # weight on utility_score (track record) in MongoRAG ranking
    "change_log": [],
}

# Average tokens per episode above which the policy tightens. The demo's
# simulated episodes use ~15-25k tokens. A real planner/executor run can use
# hundreds of thousands, so the real budget is a starting placeholder:
# calibrate it from the tokens_used of a few real runs (edit it in Atlas).
TOKEN_BUDGETS = {
    POLICY_REAL: 500_000,
    POLICY_DEMO: 50_000,
}

# Which `runs` documents feed each policy
RUNS_FILTERS = {
    POLICY_REAL: {"simulated": {"$ne": True}},
    POLICY_DEMO: {"simulated": True},
}


def default_policy(policy_id: str = POLICY_REAL) -> dict:
    return {"_id": policy_id, **DEFAULT_POLICY,
            "token_budget": TOKEN_BUDGETS[policy_id], "change_log": []}


def bootstrap_policy(policy_col: Collection, policy_id: str = POLICY_REAL):
    """Insert the default policy document if one doesn't exist yet, and add any
    fields that older documents are missing (e.g. token_budget). Safe to call every run."""
    defaults = default_policy(policy_id)
    doc = policy_col.find_one({"_id": policy_id})
    if doc is None:
        policy_col.insert_one(defaults)
        return
    missing = {k: v for k, v in defaults.items() if k not in doc}
    if missing:
        policy_col.update_one({"_id": policy_id}, {"$set": missing})


def get_policy(policy_col: Collection, policy_id: str = POLICY_REAL) -> dict:
    bootstrap_policy(policy_col, policy_id)
    return policy_col.find_one({"_id": policy_id})


def update_policy(policy_col: Collection, runs_col: Collection,
                   policy_id: str = POLICY_REAL, window: int = 10):
    """
    Look at the last `window` episodes that belong to this policy (real runs
    for "global", simulated runs for "demo") and nudge the policy if the
    harness is blowing its token budget (or comfortably under it). Pure
    metric-driven -- no LLM call needed, which keeps it fast and reliable
    to demo live.
    """
    runs_filter = {**RUNS_FILTERS[policy_id], "tokens_used": {"$gt": 0}}
    recent = list(runs_col.find(runs_filter).sort("timestamp", -1).limit(window))
    if not recent:
        return None

    avg_tokens = sum(r["tokens_used"] for r in recent) / len(recent)
    policy = get_policy(policy_col, policy_id)
    token_budget = policy["token_budget"]
    changes = {}

    if avg_tokens > token_budget:
        changes["truncate_content_chars"] = max(5000, policy["truncate_content_chars"] - 5000)
        changes["rag_inject_every"] = max(1, policy["rag_inject_every"] - 1)
    elif avg_tokens < token_budget * 0.4 and policy["truncate_content_chars"] < 25000:
        # comfortably under budget -- relax back toward the defaults
        changes["truncate_content_chars"] = min(25000, policy["truncate_content_chars"] + 5000)

    # Nothing actually moved (e.g. already at the floor): don't bump the version
    changes = {k: v for k, v in changes.items() if policy[k] != v}

    if changes:
        changes["version"] = policy["version"] + 1
        policy_col.update_one({"_id": policy_id}, {
            "$set": changes,
            "$push": {"change_log": {
                "from_version": policy["version"],
                "reason": f"avg_tokens={avg_tokens:.0f} over last {len(recent)} episodes "
                          f"(budget {token_budget})",
                "at": time.time(),
            }},
        })
        return changes
    return None
