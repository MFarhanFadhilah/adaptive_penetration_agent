"""
Self-RAG Module for CTF Solving.

Implements the Self-RAG pattern (Asai et al. 2023):
  1. RETRIEVE: decide whether to retrieve based on context
  2. ISREL:    score relevance of each retrieved document
  3. ISSUP:    judge whether the doc supports the generation
  4. ISUSE:    determine utility of using the hint

The LLM (Gemini) critiques each retrieved passage and returns only
passages that score above the relevance threshold.

All retrieval and critique events are logged for trajectory analysis.
"""
import json
import time
from typing import List, Optional, Tuple

import google.generativeai as genai

from .knowledge_base import CTFKnowledgeBase, Document


class RAGHint:
    """A knowledge hint ready to be injected into the agent context."""
    def __init__(self, doc: Document, relevance_score: float, critique: str):
        self.doc = doc
        self.relevance_score = relevance_score
        self.critique = critique

    def format(self) -> str:
        return (
            f"[RAG HINT | {self.doc.category.upper()} | score={self.relevance_score:.1f}/10]\n"
            f"**{self.doc.title}**\n"
            f"{self.doc.content}"
        )

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc.id,
            "doc_title": self.doc.title,
            "category": self.doc.category,
            "relevance_score": self.relevance_score,
            "critique": self.critique,
            "content_preview": self.doc.content[:200],
        }


class SelfRAG:
    """
    Self-RAG: Retrieval-Augmented Generation with self-reflection.

    At each retrieval call:
      1. BM25 search returns top_k candidate documents.
      2. For each candidate, the LLM scores relevance 0-10.
      3. Only documents scoring >= relevance_threshold are returned as hints.
      4. All decisions are logged for trajectory analysis.
    """

    def __init__(
        self,
        knowledge_base: CTFKnowledgeBase,
        api_key: str,
        model: str = "gemini-2.5-flash",
        relevance_threshold: float = 5.0,
        top_k: int = 3,
        max_hints_per_round: int = 2,
    ):
        self.knowledge_base = knowledge_base
        self.api_key = api_key
        self.model_name = model
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k
        self.max_hints_per_round = max_hints_per_round

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=self.model_name)

        # Full retrieval log for trajectory analysis
        self.retrieval_log: List[dict] = []

    # ------------------------------------------------------------------
    # Token 1: [Retrieve] – should we retrieve?
    # ------------------------------------------------------------------
    def should_retrieve(self, query: str, current_round: int) -> bool:
        """
        Heuristic: retrieve at round 1 (always), and every 3 rounds after.
        Could be replaced with an LLM call for full Self-RAG compliance.
        """
        return current_round == 1 or current_round % 3 == 0

    # ------------------------------------------------------------------
    # Token 2: [IsREL] – is a retrieved passage relevant?
    # ------------------------------------------------------------------
    def _critique_relevance(self, query: str, doc: Document) -> Tuple[float, str]:
        """
        Use Gemini to score how relevant a document is to the CTF query.
        Returns (score 0-10, short critique string).
        """
        prompt = (
            "You are a CTF knowledge assessor. Given a challenge context and a knowledge document, "
            "score the document's relevance from 0 (completely irrelevant) to 10 (highly relevant and directly applicable).\n\n"
            f"Challenge context:\n{query}\n\n"
            f"Knowledge document title: {doc.title}\n"
            f"Knowledge document content:\n{doc.content[:600]}\n\n"
            "Respond with JSON only in this format: "
            '{"score": <0-10 float>, "reason": "<one sentence>"}'
        )
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, max_output_tokens=128
                ),
            )
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            score = float(data.get("score", 5.0))
            reason = str(data.get("reason", ""))
            return score, reason
        except Exception as e:
            # Fallback: return neutral score
            return 5.0, f"[critique error: {e}]"

    # ------------------------------------------------------------------
    # Main RAG pipeline
    # ------------------------------------------------------------------
    def get_hints(
        self,
        challenge_name: str,
        challenge_category: str,
        challenge_description: str,
        current_plan: str = "",
        current_round: int = 1,
    ) -> List[RAGHint]:
        """
        Full Self-RAG pipeline:
        1. Decide whether to retrieve.
        2. Retrieve top_k candidates.
        3. Critique each candidate.
        4. Return hints above threshold.
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
            "retrieve_decision": False,
            "candidates": [],
            "injected_hints": [],
            "timestamp": time.time(),
        }

        if not self.should_retrieve(query, current_round):
            self.retrieval_log.append(log_entry)
            return []

        log_entry["retrieve_decision"] = True

        # Step 1: BM25 retrieval
        candidates = self.knowledge_base.search(
            query, top_k=self.top_k, category=challenge_category
        )

        hints: List[RAGHint] = []
        for doc in candidates:
            # Step 2: LLM critique
            score, critique = self._critique_relevance(query, doc)
            candidate_log = {
                "doc_id": doc.id,
                "doc_title": doc.title,
                "score": score,
                "critique": critique,
                "accepted": score >= self.relevance_threshold,
            }
            log_entry["candidates"].append(candidate_log)

            if score >= self.relevance_threshold:
                hint = RAGHint(doc=doc, relevance_score=score, critique=critique)
                hints.append(hint)

        # Limit per round
        hints = hints[: self.max_hints_per_round]
        log_entry["injected_hints"] = [h.to_dict() for h in hints]
        self.retrieval_log.append(log_entry)
        return hints

    def format_hints_for_injection(self, hints: List[RAGHint]) -> str:
        """Format a list of hints as a block to inject into planner context."""
        if not hints:
            return ""
        parts = ["=== SELF-RAG KNOWLEDGE HINTS (self-critiqued, relevant passages) ==="]
        for i, h in enumerate(hints, 1):
            parts.append(f"\n[Hint {i}/relevance={h.relevance_score:.1f}] {h.doc.title}")
            parts.append(h.doc.content)
        parts.append("=== END HINTS ===")
        return "\n".join(parts)

    def dump_log(self) -> List[dict]:
        return self.retrieval_log
