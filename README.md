# semcache: Semantic Cache for LLM Applications

A cache that returns a stored LLM response when a new query *means the same thing* as an earlier one, even if the wording differs. This cuts API cost and latency for repetitive traffic such as support bots and FAQ assistants.

## How it works

```
query ──► embed ──► cosine similarity vs. cached vectors (same namespace)
                      │
          score >= threshold ? ── yes ──► return cached response  (hit)
                      │
                      no ──► call LLM ──► store (vector, response, TTL) ──► return (miss)
```

## Features

- Cosine-similarity lookup over normalised embeddings (vectorised NumPy matmul)
- Configurable similarity threshold, per-entry TTL and LRU eviction (`max_entries`)
- Namespaces to isolate tenants, models or prompt templates
- Thread-safe (`RLock`), JSON persistence (`save` / `load`)
- Pluggable embedders: offline `HashingEmbedder` (word + bigram + char n-gram hashing) and an optional `SentenceTransformerEmbedder` for neural embeddings
- FastAPI service: `/query`, `/store`, `/chat` (cache-aside), `/stats`, `DELETE /cache`
- Metrics: hits, misses, hit rate, evictions, expirations
- Test suite (9 tests) and a reproducible benchmark

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python benchmark.py
uvicorn semcache.server:app --reload
```

```python
from semcache import SemanticCache

cache = SemanticCache(threshold=0.8, ttl=3600)
answer, was_cached, score = cache.get_or_compute("How do I reset my password?", call_llm)
answer, was_cached, score = cache.get_or_compute("How can I reset my password", call_llm)  # hit
```

## Benchmark results (default offline embedder)

Dataset: 8 canonical support questions, 24 paraphrases (should hit), 8 different-intent questions that share vocabulary (should miss).

| Threshold | Paraphrase hit rate | False-positive rate |
|-----------|--------------------:|--------------------:|
| 0.5 | 87.5% | 37.5% |
| 0.6 | 75.0% | 12.5% |
| 0.7 | 54.2% | 12.5% |
| **0.8** | **45.8%** | **0.0%** |
| 0.9 | 25.0% | 0.0% |

- Lookup latency: mean 0.17 ms at 1,000 entries; mean 1.23 ms (p95 1.31 ms) at 10,000 entries.
- The threshold sweep shows the core trade-off: a lower threshold raises hits but returns wrong answers.
- The hashing embedder is lexical, so hit rate is modest. Swapping in `SentenceTransformerEmbedder` is the main upgrade path, and the benchmark lets you measure the gain.

## Design decisions and limitations

- **Correctness over hit rate.** The default threshold is the highest-recall setting with zero false positives on the benchmark.
- **Brute-force search.** Exact and fast up to ~10^5 entries; beyond that, swap in FAISS/HNSW.
- **Known limitations.** Negation and intent flips ("enable" vs "disable") are the hardest failure mode for any embedding cache. Namespaces and TTL limit the blast radius.

## Ideas to extend

1. Neural embeddings, then re-run the benchmark and compare.
2. Redis or SQLite backend for multi-process deployments.
3. Approximate nearest neighbour index (FAISS/HNSW) and a latency chart.
4. Re-ranking or an LLM-judge verifier on borderline scores (0.7 to 0.8).
5. Prometheus metrics and Docker packaging.

## Resume bullets (adjust once you've extended it)

**Semantic Cache for LLM Applications** | Python, NumPy, FastAPI, pytest

- Built a semantic caching layer that serves stored LLM responses for paraphrased queries using cosine-similarity search over text embeddings, with configurable thresholds, TTL expiry, LRU eviction and per-tenant namespaces.
- Benchmarked threshold trade-offs on a labelled paraphrase set and selected a 0.8 operating point that avoided ~46% of LLM calls with 0% false positives; lookups take ~1.2 ms at 10k cached entries.
- Exposed the cache as a thread-safe FastAPI service (cache-aside `/chat`, `/stats`, invalidation) with JSON persistence and a 9-test pytest suite; designed a pluggable embedder interface for swapping in neural models.
- Profiled and removed an O(n) matrix rebuild on cache hits, improving 10k-entry lookup latency from ~8 ms to ~1.2 ms.
