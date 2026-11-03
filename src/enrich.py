"""Index-time enrichment: ask an LLM which questions each chunk answers, and index those alongside it.

Bridges vocabulary mismatch ("what's your return window?" vs a chunk saying "within 30 days") at no
query-time cost. Questions are generated from the chunk text only, never from the eval set, and cached
in a committed file keyed by chunk text, so a fresh clone rebuilds the same index and only changed
chunks cost an API call.
"""
import hashlib
import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .ingest import Chunk

load_dotenv()

MODEL = os.getenv("ENRICH_MODEL", "claude-haiku-5-5")
CACHE_PATH = Path("data/derived/chunk_questions.json")

PROMPT = """Below is one chunk of Alo Yoga's {kind} knowledge base. Write {n} short questions a shopper might type that this chunk answers.
Rules: vary the wording and include everyday phrasing, not just the chunk's own terms; only ask what the chunk actually answers; no customer names or order details.

{chunk}"""

TOOL = {
    "name": "questions",
    "description": "Record the questions this chunk answers.",
    "input_schema": {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                     "required": ["questions"]},
}


def enrich(chunks: list[Chunk], n: int = 4) -> list[Chunk]:
    """Append generated questions to each chunk's embedded text and record them in its metadata."""
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    client = None
    try:
        for c in chunks:
            key = hashlib.sha256(c.text.encode()).hexdigest()[:16]
            if key not in cache:
                client = client or anthropic.Anthropic()
                resp = client.messages.create(
                    model=MODEL, max_tokens=300, tools=[TOOL], tool_choice={"type": "tool", "name": "questions"},
                    messages=[{"role": "user", "content": PROMPT.format(
                        kind="policy" if c.domain == "policy" else "product catalog", n=n, chunk=c.text)}])
                cache[key] = next(b.input["questions"] for b in resp.content if b.type == "tool_use")[:n]
            c.embed_text += "\n\nQuestions this answers:\n" + "\n".join(f"- {q}" for q in cache[key])
            c.meta = {**c.meta, "questions": cache[key]}
    finally:  # keep whatever was generated, even if a later call failed
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return chunks
