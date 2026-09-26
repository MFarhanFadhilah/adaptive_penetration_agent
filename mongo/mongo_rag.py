"""
Mongo-RAG Module for CTF Solving.

Uses MongoDB Atlas Vector Search (AutoEmbed, voyage-4) over a `semantic_memory`
collection of knowledge documents. AutoEmbed means Atlas embeds both the
documents' `content` field and the query text server-side -- the app never
computes or stores its own embeddings (see mongo/setup_atlas.py, which
creates the `autoembed_index` this module queries).

Retrieval is blended with outcome-weighted ranking so hints that have
previously led to a solve are preferred over ones that haven't:

    blended_score = alpha * vector_similarity + beta * utility_score

`vector_similarity` comes from Atlas's $vectorSearch `searchScore`.
`utility_score` is a field maintained on each document by the outcome-tracking
Atlas Trigger that watches the `runs` collection (see
MONGODB_FINAL_GUIDE.md section 1.3): every run that used a doc bumps
`times_retrieved`, and every solved run also bumps `times_led_to_success` and
recomputes `utility_score`.

All retrieval events are logged for trajectory analysis, mirroring
nyuctf_multiagent/rag/self_rag.py and graph_rag.py.
"""
import time
from dataclasses import dataclass
from typing import List

# Name of the Atlas Vector Search index on `semantic_memory` (AutoEmbed over
# `content`, filterable by `category`). Created by mongo/setup_atlas.py.
INDEX_NAME = "autoembed_index"


def category_filter(category: str) -> dict:
    """
    $vectorSearch filter for a challenge category: matches that category's
    docs plus "general" ones (methodology/tooling hints that apply everywhere).
    """
    return {"category": {"$in": [category, "general"]}} if category else {}


@dataclass
class MongoHint:
    """A knowledge hint retrieved from Atlas, ready to be injected into the agent context."""
    doc_id: str
    title: str
    content: str
    category: str
    similarity: float
    utility_score: float
    blended_score: float

    def format(self) -> str:
        return (
            f"[Mongo-RAG HINT | {self.category.upper()} | "
            f"score={self.blended_score:.2f} (sim={self.similarity:.2f}, utility={self.utility_score:.2f})]\n"
            f"**{self.title}**\n{self.content}"
        )

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "similarity": self.similarity,
            "utility_score": self.utility_score,
            "blended_score": self.blended_score,
            "content_preview": self.content[:200],
        }


class MongoRAG:
    """
    Mongo-RAG: Atlas Vector Search (AutoEmbed) retrieval blended with
    outcome-weighted ranking, over a `semantic_memory` collection.
    """

    def __init__(
        self,
        collection,
        top_k: int = 3,
        alpha: float = 0.6,
        beta: float = 0.4,
        max_hints_per_round: int = 2,
        inject_every: int = 3,
    ):
        self.collection = collection
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.max_hints_per_round = max_hints_per_round
        self.inject_every = inject_every

        # Full retrieval log for trajectory analysis
        self.retrieval_log: List[dict] = []
        # doc_ids injected this episode, reported back in `runs.used_doc_ids`
        self.episode_injections: List[str] = []

    def get_hints(
        self,
        challenge_name: str,
        challenge_category: str,
        challenge_description: str,
        current_plan: str = "",
        current_round: int = 1,
    ) -> List[MongoHint]:
        """
        1. $vectorSearch on `semantic_memory` with the challenge context as
           the AutoEmbed query text, filtered to this category (+ "general").
        2. Blend similarity with utility_score, rank, and keep top hints.
        """
        query = (
            f"Challenge: {challenge_name} (category: {challenge_category})\n"
            f"Description: {challenge_description}\n"
        )
        if current_plan:
            query += f"Current plan/context: {current_plan[:400]}"

        log_entry = {
            "round": current_round,
            "query_preview": query[:300],
            "candidates": [],
            "injected_hints": [],
            "timestamp": time.time(),
        }

        pipeline = [
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "content",
                    "query": query,
                    "numCandidates": max(50, self.top_k * 10),
                    "limit": self.top_k,
                    "filter": category_filter(challenge_category),
                }
            },
            {
                "$project": {
                    "doc_id": 1,
                    "title": 1,
                    "content": 1,
                    "category": 1,
                    "utility_score": 1,
                    "similarity": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            candidates = list(self.collection.aggregate(pipeline))
        except Exception as e:
            log_entry["error"] = f"vector search failed: {e}"
            self.retrieval_log.append(log_entry)
            return []

        hints: List[MongoHint] = []
        for doc in candidates:
            similarity = float(doc.get("similarity", 0.0))
            utility_score = float(doc.get("utility_score", 0.5))
            hint = MongoHint(
                doc_id=doc.get("doc_id", str(doc.get("_id"))),
                title=doc.get("title", "untitled"),
                content=doc.get("content", ""),
                category=doc.get("category", challenge_category),
                similarity=similarity,
                utility_score=utility_score,
                blended_score=self.alpha * similarity + self.beta * utility_score,
            )
            hints.append(hint)
            log_entry["candidates"].append(hint.to_dict())

        hints.sort(key=lambda h: -h.blended_score)
        hints = hints[: self.max_hints_per_round]
        log_entry["injected_hints"] = [h.to_dict() for h in hints]
        self.retrieval_log.append(log_entry)
        self.episode_injections.extend(h.doc_id for h in hints)
        return hints

    def format_hints_for_injection(self, hints: List[MongoHint]) -> str:
        """Format a list of hints as a block to inject into planner context."""
        if not hints:
            return ""
        parts = ["=== MONGO-RAG KNOWLEDGE HINTS (Atlas Vector Search, outcome-weighted) ==="]
        for i, h in enumerate(hints, 1):
            parts.append(f"\n[Hint {i}/score={h.blended_score:.2f}] {h.title}")
            parts.append(h.content)
        parts.append("=== END HINTS ===")
        return "\n".join(parts)

    def dump_log(self) -> List[dict]:
        return self.retrieval_log
