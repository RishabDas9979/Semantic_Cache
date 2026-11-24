"""HTTP API: `uvicorn semcache.server:app`.

POST /query   {query, namespace?}            -> cached response or {"cached": false}
POST /store   {query, response, namespace?}  -> store an LLM answer
POST /chat    {query, namespace?}            -> cache-aside: calls LLM on miss
GET  /stats                                  -> hit rate, evictions, latency saved
DELETE /cache?namespace=...                  -> invalidate
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .cache import SemanticCache

app = FastAPI(title="semcache", version="1.0.0")
cache = SemanticCache(
    threshold=float(os.getenv("SEMCACHE_THRESHOLD", "0.8")),
    ttl=float(os.getenv("SEMCACHE_TTL", "3600")),
    max_entries=int(os.getenv("SEMCACHE_MAX_ENTRIES", "10000")),
)


class QueryIn(BaseModel):
    query: str
    namespace: str = "default"


class StoreIn(QueryIn):
    response: str
    ttl: Optional[float] = None


def call_llm(prompt: str) -> str:
    """Replace with a real provider call (OpenAI, Anthropic, ...)."""
    time.sleep(float(os.getenv("SEMCACHE_FAKE_LLM_LATENCY", "0.8")))
    return f"[LLM answer to: {prompt}]"


@app.post("/query")
def query(body: QueryIn):
    m = cache.lookup(body.query, body.namespace)
    if not m:
        return {"cached": False}
    return {"cached": True, "response": m.entry.response, "similarity": round(m.score, 4),
            "matched_query": m.entry.query}


@app.post("/store")
def store(body: StoreIn):
    e = cache.store(body.query, body.response, body.namespace, body.ttl)
    return {"id": e.id}


@app.post("/chat")
def chat(body: QueryIn):
    t0 = time.perf_counter()
    m = cache.lookup(body.query, body.namespace)
    if m:
        return {"cached": True, "response": m.entry.response, "similarity": round(m.score, 4),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    resp = call_llm(body.query)
    cache.store(body.query, resp, body.namespace)
    return {"cached": False, "response": resp, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/stats")
def stats():
    return {"entries": len(cache), **cache.stats.as_dict()}


@app.delete("/cache")
def clear(namespace: Optional[str] = None):
    return {"cleared": cache.clear(namespace)}
