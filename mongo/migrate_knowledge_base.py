"""
One-off migration: load nyuctf_multiagent/rag/ctf_knowledge_base.json,
embed each document with Gemini, and upsert into MongoDB Atlas.

This does not touch the live agent (self_rag.py / knowledge_base.py) --
it's a standalone script to get the same knowledge base into Atlas so we
can test Vector Search against it before wiring it into the harness.

Usage:
    python3 mongo/migrate_knowledge_base.py --keys keys.cfg
"""
import argparse
import json
import time
from pathlib import Path

import google.generativeai as genai
from pymongo import MongoClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyuctf_multiagent.utils import APIKeys

EMBED_MODEL = "models/text-embedding-004"
KB_PATH = Path(__file__).resolve().parent.parent / "nyuctf_multiagent" / "rag" / "ctf_knowledge_base.json"


def embed(text: str, task_type: str = "retrieval_document"):
    resp = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
    return resp["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", default="keys.cfg", help="Path to keys.cfg (needs GEMINI= and MONGODB_URI=)")
    parser.add_argument("--db", default="pentest_memory")
    parser.add_argument("--collection", default="semantic_memory")
    args = parser.parse_args()

    keys = APIKeys(args.keys)
    genai.configure(api_key=keys["GEMINI"])
    mongo = MongoClient(keys["MONGODB_URI"])
    collection = mongo[args.db][args.collection]

    with open(KB_PATH) as f:
        data = json.load(f)

    inserted, skipped, dim = 0, 0, None
    for doc in data["documents"]:
        if collection.find_one({"doc_id": doc["id"]}):
            skipped += 1
            continue
        full_text = f"{doc['title']}. {doc['content']} Keywords: {' '.join(doc['keywords'])}"
        vector = embed(full_text)
        dim = len(vector)
        collection.insert_one({
            "doc_id": doc["id"],
            "category": doc["category"],
            "title": doc["title"],
            "content": doc["content"],
            "keywords": doc["keywords"],
            "embedding": vector,
            # hard-metric fields, updated later from real episode outcomes
            "times_retrieved": 0,
            "times_led_to_success": 0,
            "utility_score": 0.5,
        })
        inserted += 1
        print(f"embedded: {doc['id']} ({doc['title']})")
        time.sleep(0.2)  # gentle on the embedding API rate limit

    print(f"\nDone. inserted={inserted} skipped(existing)={skipped} embedding_dim={dim or 'n/a (nothing new inserted)'}")
    print("If embedding_dim is shown above, use that exact number as numDimensions when creating the Atlas Vector Search index.")


if __name__ == "__main__":
    main()
