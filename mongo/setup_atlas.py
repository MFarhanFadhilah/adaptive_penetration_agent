"""
One-command setup and health check for the `pentest_memory` database on Atlas.

Idempotent -- safe to run on a fresh cluster or on one that's already set up:
  1. creates the collections (semantic_memory, runs, episodic_log, policy)
  2. loads the knowledge docs from mongo/kb_array_for_import.json
     (existing docs, and their learned utility scores, are left alone)
  3. creates the regular indexes the harness queries by
  4. creates the `autoembed_index` Vector Search index (AutoEmbed, voyage-4,
     `category` filter) and waits until it's queryable
  5. creates the two policy documents ("global" for the real agent, "demo")
  6. checks Vector Search returns on-topic results
  7. checks the outcome-tracking Atlas Trigger on `runs` really fires, by
     inserting a test run and restoring everything afterwards

The one thing this can't create is the Trigger itself: that needs an Atlas
admin API key, not a database user. See MONGODB_FINAL_GUIDE.md section 1.3.

Usage:
    python3 mongo/setup_atlas.py --keys keys.cfg
    python3 mongo/setup_atlas.py --keys keys.cfg --check-only
"""
import argparse
import json
import sys
import time
from pathlib import Path

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.operations import SearchIndexModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyuctf_multiagent.utils import APIKeys
from mongo.mongo_rag import INDEX_NAME, category_filter
from mongo.policy import POLICY_DEMO, POLICY_REAL, get_policy

DB_NAME = "pentest_memory"
COLLECTIONS = ["semantic_memory", "runs", "episodic_log", "policy"]
KB_PATH = Path(__file__).resolve().parent / "kb_array_for_import.json"
VECTOR_INDEX = {
    "fields": [
        {"type": "autoEmbed", "modality": "text", "path": "content", "model": "voyage-4"},
        {"type": "filter", "path": "category"},
    ]
}
TEST_DOC_ID = "gen-001"

results = []


def report(ok: bool, name: str, detail: str = ""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


def setup(db):
    print("Setting up:")
    existing = db.list_collection_names()
    for name in COLLECTIONS:
        if name not in existing:
            db.create_collection(name)
    print(f"  collections: {', '.join(COLLECTIONS)}")

    docs = json.loads(KB_PATH.read_text())
    inserted = 0
    for doc in docs:
        r = db.semantic_memory.update_one({"doc_id": doc["doc_id"]}, {"$setOnInsert": doc}, upsert=True)
        inserted += r.upserted_id is not None
    print(f"  knowledge docs: {inserted} inserted, {len(docs) - inserted} already present")

    db.semantic_memory.create_index("doc_id", unique=True)
    db.runs.create_index([("timestamp", DESCENDING)])
    db.runs.create_index([("simulated", ASCENDING), ("timestamp", DESCENDING)])
    db.runs.create_index("episode_id")
    db.episodic_log.create_index([("episode_id", ASCENDING), ("seq", ASCENDING)])
    print("  indexes: semantic_memory.doc_id, runs.timestamp, runs.simulated+timestamp, "
          "runs.episode_id, episodic_log.episode_id+seq")

    if not list(db.semantic_memory.list_search_indexes(INDEX_NAME)):
        db.semantic_memory.create_search_index(
            SearchIndexModel(definition=VECTOR_INDEX, name=INDEX_NAME, type="vectorSearch"))
        print(f"  vector index: {INDEX_NAME} created, waiting for it to build...")
    else:
        print(f"  vector index: {INDEX_NAME} already exists")
    wait_for_index(db)

    for policy_id in (POLICY_REAL, POLICY_DEMO):
        get_policy(db.policy, policy_id)
    print(f"  policies: \"{POLICY_REAL}\" and \"{POLICY_DEMO}\"")


def wait_for_index(db, timeout_s=300):
    started = time.time()
    while time.time() - started < timeout_s:
        idx = list(db.semantic_memory.list_search_indexes(INDEX_NAME))
        if idx and idx[0].get("queryable"):
            return True
        if idx and idx[0].get("status") == "FAILED":
            return False
        time.sleep(10)
    return False


def check(db, test_trigger: bool):
    print("Checking:")
    names = db.list_collection_names()
    missing = [c for c in COLLECTIONS if c not in names]
    report(not missing, "collections exist", f"missing {missing}" if missing else "")

    n_docs = db.semantic_memory.count_documents({})
    report(n_docs >= 33, "knowledge base loaded", f"{n_docs} docs")

    idx_names = {i["name"] for i in db.runs.list_indexes()} | {i["name"] for i in db.episodic_log.list_indexes()}
    report({"timestamp_-1", "episode_id_1_seq_1"} <= idx_names, "runs / episodic_log indexes")

    idx = list(db.semantic_memory.list_search_indexes(INDEX_NAME))
    ok = bool(idx) and idx[0].get("queryable")
    report(ok, f"vector index {INDEX_NAME} queryable", idx[0].get("status") if idx else "missing")
    if ok:
        fields = idx[0].get("latestDefinition", {}).get("fields", [])
        has_filter = any(f.get("type") == "filter" and f.get("path") == "category" for f in fields)
        report(has_filter, "vector index has the `category` filter field")
        hits = list(db.semantic_memory.aggregate([
            {"$vectorSearch": {"index": INDEX_NAME, "path": "content",
                               "query": "web challenge with a login form, suspect SQL injection",
                               "numCandidates": 100, "limit": 3, "filter": category_filter("web")}},
            {"$project": {"_id": 0, "doc_id": 1, "category": 1}},
        ]))
        on_topic = bool(hits) and all(h["category"] in ("web", "general") for h in hits)
        report(on_topic, "vector search returns on-topic results", ", ".join(h["doc_id"] for h in hits))

    for policy_id in (POLICY_REAL, POLICY_DEMO):
        p = db.policy.find_one({"_id": policy_id})
        report(p is not None and "token_budget" in p, f"policy \"{policy_id}\"",
               f"v{p['version']}, token_budget={p['token_budget']}" if p else "missing")

    if test_trigger:
        check_trigger(db)


def check_trigger(db, timeout_s=15):
    """Insert a run that used TEST_DOC_ID and wait for the Atlas Trigger to
    bump its counters; then put the doc back exactly as it was."""
    fields = {"_id": 0, "times_retrieved": 1, "times_led_to_success": 1, "utility_score": 1}
    before = db.semantic_memory.find_one({"doc_id": TEST_DOC_ID}, fields)
    if before is None:
        report(False, "outcome trigger fires", f"{TEST_DOC_ID} not found")
        return
    run_id = db.runs.insert_one({
        "episode_id": "setup-atlas-trigger-check", "used_doc_ids": [TEST_DOC_ID], "solved": True,
        "tokens_used": 0, "simulated": True, "timestamp": time.time(),
    }).inserted_id
    fired, waited = False, 0.0
    try:
        started = time.time()
        while time.time() - started < timeout_s:
            now = db.semantic_memory.find_one({"doc_id": TEST_DOC_ID}, fields)
            if now.get("times_retrieved", 0) > before.get("times_retrieved", 0):
                fired, waited = True, time.time() - started
                break
            time.sleep(0.5)
    finally:
        db.runs.delete_one({"_id": run_id})
        db.semantic_memory.update_one({"doc_id": TEST_DOC_ID}, {"$set": before})
    report(fired, "outcome trigger on `runs` fires",
           f"updated {TEST_DOC_ID} in {waited:.1f}s (restored)" if fired else
           f"no update within {timeout_s}s -- create/fix it per MONGODB_FINAL_GUIDE.md section 1.3")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--keys", default="keys.cfg", help="keys.cfg with MONGODB_URI=")
    parser.add_argument("--check-only", action="store_true", help="don't create anything, just check")
    parser.add_argument("--skip-trigger-test", action="store_true",
                        help="don't insert a test run to check the Atlas Trigger")
    args = parser.parse_args()

    db = MongoClient(APIKeys(args.keys)["MONGODB_URI"])[DB_NAME]
    if not args.check_only:
        setup(db)
    check(db, test_trigger=not args.skip_trigger_test)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed" + ("" if passed == len(results) else " -- see FAIL lines above"))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
