"""
CTF Knowledge Base for RAG systems.

Loads CTF knowledge documents and provides BM25/TF-IDF-based retrieval.
Documents are stored in ctf_knowledge_base.json alongside this module.
"""
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    id: str
    category: str
    title: str
    content: str
    keywords: List[str]

    def full_text(self) -> str:
        return f"{self.title}. {self.content} Keywords: {' '.join(self.keywords)}"


class CTFKnowledgeBase:
    """
    Loads and indexes CTF knowledge documents.
    Supports keyword and TF-IDF-based retrieval.
    """

    def __init__(self, kb_path: Optional[str] = None):
        if kb_path is None:
            kb_path = Path(__file__).parent / "ctf_knowledge_base.json"
        self.documents: List[Document] = []
        self._load(kb_path)
        self._build_index()

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data["documents"]:
            self.documents.append(
                Document(
                    id=doc["id"],
                    category=doc["category"],
                    title=doc["title"],
                    content=doc["content"],
                    keywords=doc["keywords"],
                )
            )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _build_index(self):
        """Build an inverted index and IDF table for BM25."""
        self._doc_tokens: List[List[str]] = []
        for doc in self.documents:
            tokens = self._tokenize(doc.full_text())
            self._doc_tokens.append(tokens)

        # IDF
        N = len(self.documents)
        df: dict = {}
        for tokens in self._doc_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}
        self._avg_dl = sum(len(t) for t in self._doc_tokens) / max(N, 1)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int,
                    k1: float = 1.5, b: float = 0.75) -> float:
        tokens = self._doc_tokens[doc_idx]
        dl = len(tokens)
        tf_map: dict = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for qt in query_tokens:
            if qt not in self._idf:
                continue
            tf = tf_map.get(qt, 0)
            idf = self._idf[qt]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_dl))
        return score

    def search(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[Document]:
        """
        Retrieve top_k most relevant documents for the query.
        Optionally filter by category.
        """
        query_tokens = self._tokenize(query)
        # Also expand with category keyword if provided
        if category:
            query_tokens += self._tokenize(category)

        scores = []
        for i, doc in enumerate(self.documents):
            if category and doc.category not in (category, "general"):
                # Allow general docs always, filter non-matching categories
                # but don't skip them entirely - just lower priority (half weight)
                score = self._bm25_score(query_tokens, i) * 0.5
            else:
                score = self._bm25_score(query_tokens, i)
            scores.append((score, i))

        scores.sort(key=lambda x: -x[0])
        results = [self.documents[i] for _, i in scores[:top_k] if _ > 0]
        return results

    def get_by_category(self, category: str) -> List[Document]:
        return [d for d in self.documents if d.category == category]

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
