"""
MongoRAG: drop-in replacement for SelfRAG (nyuctf_multiagent/rag/self_rag.py)
that retrieves from MongoDB Atlas Vector Search instead of in-memory BM25,
and ranks results by real outcome history (utility_score) instead of an
LLM's guessed relevance score.

Uses Atlas Vector Search's Automated Embedding (AutoEmbed) on the `content`
field with the voyage-4 model. Atlas embeds both the stored documents and
the query text itself -- this module never calls an embedding model or
computes a vector locally, it just passes plain query text into
`$vectorSearch`'s `query` field and lets Atlas do the rest. Confirmed
working against a live Atlas cluster on 2026-09-24.

Exposes the same shape SelfRAG does (get_hints / format_hints_for_injection)
so rag_agent.py can swap one for the other with minimal changes.
"""
import time
from typing import List

from pymongo.collection import Collection

INDEX_NAME = "autoembed_index"

# NYU CTF challenge categories -> knowledge-base `category` values. "general"
# docs (methodology, common tools) apply to every category, so they're always
# allowed; "misc" challenges only get those.
KB_CATEGORY = {
    "web": "web",
    "pwn": "pwn",
    "rev": "rev",
    "crypto": "cry",
    "forensics": "for",
    "misc": None,
}
GENERAL_CATEGORY = "general"


def category_filter(challenge_category: str):
    """Pre-filter for $vectorSearch (the index has `category` as a filter field),
    or None to search everything when the category is unknown."""
    key = (challenge_category or "").strip().lower()
    if key not in KB_CATEGORY:
        return None
    allowed = [GENERAL_CATEGORY] + ([KB_CATEGORY[key]] if KB_CATEGORY[key] else [])
    return {"category": {"$in": allowed}}


class MongoHint:
    def __init__(self, doc: dict, vscore: float, utility: float, final_score: float):
        self.doc = doc
        self.vscore = vscore
        self.utility = utility
        self.final_score = final_score

    def format(self) -> str:
        return (
            f"[Mongo-RAG HINT | {self.doc['category'].upper()} | "
            f"similarity={self.vscore:.2f} utility={self.utility:.2f}]\n"
            f"**{self.doc['title']}**\n{self.doc['content']}"
        )

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc["doc_id"],
            "doc_title": self.doc["title"],
            "category": self.doc["category"],
            "vscore": self.vscore,
            "utility_score": self.utility,
            "final_score": self.final_score,
            "content_preview": self.doc["content"][:200],
        }


class MongoRAG:
    """Self-RAG-compatible retriever backed by MongoDB Atlas Vector Search."""

    def __init__(self, collection: Collection, top_k: int = 3, alpha: float = 0.6,
                 beta: float = 0.4, max_hints_per_round: int = 2, inject_every: int = 3):
        self.collection = collection
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.max_hints_per_round = max_hints_per_round
        self.inject_every = inject_every
        self.retrieval_log: List[dict] = []
        # doc_ids handed to the agent this episode -- feed this into record_outcome() later
        self.episode_injections: List[str] = []

    def should_retrieve(self, current_round: int) -> bool:
        return current_round == 1 or current_round % self.inject_every == 0

    def get_hints(self, challenge_name, challenge_category, challenge_description,
                  current_plan: str = "", current_round: int = 1) -> List[MongoHint]:
        log_entry = {
            "round": current_round, "retrieve_decision": False,
            "candidates": [], "injected_hints": [], "timestamp": time.time(),
        }
        if not self.should_retrieve(current_round):
            self.retrieval_log.append(log_entry)
            return []
        log_entry["retrieve_decision"] = True

        query = f"{challenge_category} {challenge_name} {challenge_description} {current_plan[:300]}"

        vector_search = {
            "index": INDEX_NAME, "path": "content", "query": query,
            "numCandidates": 100, "limit": self.top_k * 3,
        }
        # Only this category's techniques (plus general ones), so off-topic hints
        # neither waste prompt tokens nor get credit/blame for this episode's outcome
        search_filter = category_filter(challenge_category)
        if search_filter:
            vector_search["filter"] = search_filter
        log_entry["category_filter"] = search_filter

        results = list(self.collection.aggregate([
            {"$vectorSearch": vector_search},
            {"$addFields": {"vscore": {"$meta": "vectorSearchScore"}}},
            {"$addFields": {"final_score": {"$add": [
                {"$multiply": ["$vscore", self.alpha]},
                {"$multiply": ["$utility_score", self.beta]},
            ]}}},
            {"$sort": {"final_score": -1}},
            {"$limit": self.top_k},
        ]))

        hints = [MongoHint(r, r["vscore"], r["utility_score"], r["final_score"]) for r in results]
        hints = hints[: self.max_hints_per_round]

        log_entry["candidates"] = [h.to_dict() for h in hints]
        log_entry["injected_hints"] = [h.to_dict() for h in hints]
        self.retrieval_log.append(log_entry)
        self.episode_injections.extend(h.doc["doc_id"] for h in hints)
        return hints

    def format_hints_for_injection(self, hints: List[MongoHint]) -> str:
        if not hints:
            return ""
        parts = ["=== MONGO-RAG KNOWLEDGE HINTS (vector search + outcome-weighted) ==="]
        for i, h in enumerate(hints, 1):
            parts.append(f"\n[Hint {i}/final_score={h.final_score:.2f}] {h.doc['title']}")
            parts.append(h.doc["content"])
        parts.append("=== END HINTS ===")
        return "\n".join(parts)

    def dump_log(self) -> List[dict]:
        return self.retrieval_log


def record_outcome(collection: Collection, injected_doc_ids: List[str], solved: bool):
    """
    Update times_retrieved / times_led_to_success / utility_score for every
    memory that was actually shown to the agent this episode.

    Call this once, at the end of run(), once self.environment.solved is known.
    This is the piece that makes the memory self-evolving: every episode changes
    what future episodes will prefer to retrieve, based on real outcomes rather
    than an LLM's guess.
    """
    for doc_id in set(injected_doc_ids):
        collection.update_one({"doc_id": doc_id}, {"$inc": {
            "times_retrieved": 1,
            "times_led_to_success": 1 if solved else 0,
        }})
        doc = collection.find_one({"doc_id": doc_id})
        if doc is None:
            continue
        # Laplace smoothing so one lucky/unlucky use can't swing the score wildly
        new_score = (doc["times_led_to_success"] + 1) / (doc["times_retrieved"] + 2)
        collection.update_one({"doc_id": doc_id}, {"$set": {"utility_score": new_score}})
