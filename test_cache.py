import os
import tempfile

import pytest

from semcache import SemanticCache


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_exact_and_paraphrase_hit():
    c = SemanticCache(threshold=0.7)
    c.store("How do I reset my password?", "Click 'Forgot password'.")
    assert c.lookup("How do I reset my password?").score > 0.99
    assert c.lookup("How can I reset my password") is not None


def test_unrelated_query_misses():
    c = SemanticCache(threshold=0.8)
    c.store("How do I reset my password?", "A")
    assert c.lookup("What is the capital of France?") is None
    assert c.stats.misses == 1


def test_namespace_isolation():
    c = SemanticCache(threshold=0.7)
    c.store("What is your refund policy?", "A", namespace="tenant1")
    assert c.lookup("What is your refund policy?", namespace="tenant2") is None
    assert c.lookup("What is your refund policy?", namespace="tenant1") is not None


def test_ttl_expiry():
    clock = FakeClock()
    c = SemanticCache(ttl=60, clock=clock)
    c.store("hello world question", "A")
    assert c.lookup("hello world question") is not None
    clock.t += 61
    assert c.lookup("hello world question") is None
    assert c.stats.expired == 1


def test_lru_eviction_keeps_recently_used():
    c = SemanticCache(max_entries=2, threshold=0.9)
    c.store("alpha topic apples", "1")
    c.store("beta topic bananas", "2")
    c.lookup("alpha topic apples")  # touch alpha -> beta is now LRU
    c.store("gamma topic grapes", "3")
    assert c.lookup("beta topic bananas") is None
    assert c.lookup("alpha topic apples") is not None
    assert c.stats.evictions == 1


def test_get_or_compute_calls_llm_once():
    c = SemanticCache(threshold=0.7)
    calls = []

    def llm(q):
        calls.append(q)
        return "answer"

    r1 = c.get_or_compute("How do I cancel my subscription?", llm)
    r2 = c.get_or_compute("How can I cancel my subscription", llm)
    assert r1[1] is False and r2[1] is True
    assert len(calls) == 1


def test_persistence_roundtrip():
    c = SemanticCache(threshold=0.7)
    c.store("How long does shipping take?", "3-5 days")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cache.json")
        c.save(p)
        c2 = SemanticCache(threshold=0.7)
        assert c2.load(p) == 1
        assert c2.lookup("How long does shipping take?").entry.response == "3-5 days"


def test_clear_by_namespace():
    c = SemanticCache()
    c.store("a b c", "1", namespace="x")
    c.store("d e f", "2", namespace="y")
    assert c.clear("x") == 1
    assert len(c) == 1


def test_api_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    os.environ["SEMCACHE_FAKE_LLM_LATENCY"] = "0"
    from semcache.server import app
    client = TestClient(app)
    first = client.post("/chat", json={"query": "How do I reset my password?"}).json()
    second = client.post("/chat", json={"query": "How can I reset my password"}).json()
    assert first["cached"] is False and second["cached"] is True
    assert client.get("/stats").json()["hits"] >= 1
