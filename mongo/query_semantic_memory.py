"""
Test script: given a CTF-style query, run Atlas Vector Search against
the semantic_memory collection and print the top matches, ranked by a
blend of semantic similarity and utility_score (hard-metric track record).

Uses Atlas Vector Search's Automated Embedding (AutoEmbed, voyage-4, on the
`content` field) -- Atlas embeds the query text itself, so this script
never calls an embedding model or computes a vector locally.

Usage:
    python3 mongo/query_semantic_memory.py --keys keys.cfg \\
        --query "web challenge with a login form, suspect SQL injection"
"""
import argparse
import sys
from pathlib import Path

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyuctf_multiagent.utils import APIKeys
from mongo.mongo_rag import INDEX_NAME, category_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", default="keys.cfg")
    parser.add_argument("--db", default="pentest_memory")
    parser.add_argument("--collection", default="semantic_memory")
    parser.add_argument("--query", required=True)
    parser.add_argument("--category", default=None,
                        help="challenge category (web, pwn, rev, crypto, forensics, misc) to restrict results to")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.6, help="weight on semantic similarity")
    parser.add_argument("--beta", type=float, default=0.4, help="weight on utility_score")
    args = parser.parse_args()

    keys = APIKeys(args.keys)
    mongo = MongoClient(keys["MONGODB_URI"])
    collection = mongo[args.db][args.collection]

    vector_search = {
        "index": INDEX_NAME,
        "path": "content",
        "query": args.query,
        "numCandidates": 100,
        "limit": args.top_k * 3,  # overfetch, rerank below
    }
    if args.category and category_filter(args.category):
        vector_search["filter"] = category_filter(args.category)

    results = collection.aggregate([
        {"$vectorSearch": vector_search},
        {"$addFields": {"vscore": {"$meta": "vectorSearchScore"}}},
        {"$addFields": {"final_score": {
            "$add": [
                {"$multiply": ["$vscore", args.alpha]},
                {"$multiply": ["$utility_score", args.beta]},
            ]
        }}},
        {"$sort": {"final_score": -1}},
        {"$limit": args.top_k},
        {"$project": {"_id": 0, "doc_id": 1, "title": 1, "category": 1, "vscore": 1, "utility_score": 1, "final_score": 1}},
    ])

    print(f"Query: {args.query}\n")
    for i, r in enumerate(results, 1):
        print(
            f"{i}. [{r['category']}] {r['title']}  "
            f"(similarity={r['vscore']:.3f}, utility={r['utility_score']:.2f}, final={r['final_score']:.3f})  "
            f"id={r['doc_id']}"
        )


if __name__ == "__main__":
    main()
