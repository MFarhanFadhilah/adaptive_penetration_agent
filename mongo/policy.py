"""
Self-tuning policy for Mongo-RAG.

The `policy` collection holds one evolving document per profile:
  - POLICY_REAL ("global"): the actual harness reads this at startup (see
    run_rag.py's `resolve_setting` precedence: CLI > policy > YAML > default).
  - POLICY_DEMO ("demo"): a separate, lower-stakes profile the demo webapp can
    tune independently without touching the real harness's settings.

After each episode, update_policy() looks at the rolling average token usage
of recent `runs` against the policy's `token_budget` and nudges the policy:
  - Usage comfortably over budget -> tighten: smaller top_k and
    truncate_content_chars (less context per hint, less noise) and inject
    hints more often, to try to bring cost down.
  - Usage comfortably under budget -> relax back toward the defaults.

This gives the harness closed-loop feedback without a human editing YAML
between runs, and is exactly what mongo/setup_atlas.py's health check reads
back (policy["token_budget"], policy["version"]).
"""
import time

POLICY_REAL = "global"
POLICY_DEMO = "demo"

DEFAULT_POLICY = {
    "version": 1,
    "token_budget": 40_000,
    "alpha": 0.6,
    "beta": 0.4,
    "rag_top_k": 3,
    "rag_inject_every": 3,
    "truncate_content_chars": None,
}

# Runs considered when computing the rolling average
WINDOW = 10


def get_policy(policy_collection, policy_id: str = POLICY_REAL) -> dict:
    """Return the policy document for `policy_id`, creating the default one if absent."""
    doc = policy_collection.find_one({"_id": policy_id})
    if doc is None:
        doc = {"_id": policy_id, **DEFAULT_POLICY, "updated_at": time.time()}
        policy_collection.insert_one(doc)
    return doc


def update_policy(policy_collection, runs_collection, policy_id: str = POLICY_REAL) -> dict:
    """
    React to the last WINDOW runs' token usage against `token_budget` and
    adjust the policy in place. Returns the fields that changed (empty dict
    if nothing changed).
    """
    policy = get_policy(policy_collection, policy_id)
    token_budget = policy.get("token_budget", DEFAULT_POLICY["token_budget"])
    recent = list(
        runs_collection.find({}, {"tokens_used": 1}).sort("timestamp", -1).limit(WINDOW)
    )
    if not recent:
        return {}

    avg_tokens = sum(r.get("tokens_used", 0) for r in recent) / len(recent)

    changes = {}
    if avg_tokens > token_budget * 1.2:
        new_top_k = max(1, policy.get("rag_top_k", 3) - 1)
        new_inject_every = max(1, policy.get("rag_inject_every", 3) - 1)
        new_truncate = min(policy.get("truncate_content_chars") or 4000, 2000)
        if new_top_k != policy.get("rag_top_k"):
            changes["rag_top_k"] = new_top_k
        if new_inject_every != policy.get("rag_inject_every"):
            changes["rag_inject_every"] = new_inject_every
        if new_truncate != policy.get("truncate_content_chars"):
            changes["truncate_content_chars"] = new_truncate
    elif avg_tokens < token_budget * 0.5:
        if policy.get("rag_top_k", 3) < DEFAULT_POLICY["rag_top_k"]:
            changes["rag_top_k"] = DEFAULT_POLICY["rag_top_k"]
        if policy.get("rag_inject_every", 3) < DEFAULT_POLICY["rag_inject_every"]:
            changes["rag_inject_every"] = DEFAULT_POLICY["rag_inject_every"]
        if policy.get("truncate_content_chars") is not None:
            changes["truncate_content_chars"] = None

    if changes:
        changes["version"] = policy.get("version", 1) + 1
        changes["updated_at"] = time.time()
        policy_collection.update_one({"_id": policy_id}, {"$set": changes})

    return changes
