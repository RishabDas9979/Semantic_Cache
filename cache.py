"""Semantic cache: return a stored response when a new query is similar enough
to a previously answered one, skipping an expensive LLM call."""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .embedders import Embedder, HashingEmbedder


@dataclass
class Entry:
    id: int
    query: str
    response: str
    vector: np.ndarray
    namespace: str
    created_at: float
    expires_at: Optional[float]
    hits: int = 0


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    saved_seconds: float = 0.0
    lookup_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["hit_rate"] = round(self.hit_rate, 4)
        return d


@dataclass
class Match:
    entry: Entry
    score: float


class SemanticCache:
    """Thread-safe semantic cache with cosine-similarity lookup.

    - threshold: minimum cosine similarity to count as a hit
    - ttl: seconds before an entry expires (None = never)
    - max_entries: LRU eviction beyond this size
    - namespaces isolate tenants / models / prompt templates
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        threshold: float = 0.8,
        ttl: Optional[float] = None,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.time,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: "OrderedDict[int, Entry]" = OrderedDict()  # LRU order
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._ids: list[int] = []
        self._next_id = 0
        self.stats = Stats()

    # ---------- internals ----------
    def _rebuild(self) -> None:
        self._ids = list(self._entries.keys())
        if self._ids:
            self._matrix = np.stack([self._entries[i].vector for i in self._ids])
        else:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def _purge_expired(self) -> None:
        now = self._clock()
        dead = [i for i, e in self._entries.items() if e.expires_at is not None and e.expires_at <= now]
        for i in dead:
            del self._entries[i]
        if dead:
            self.stats.expired += len(dead)
            self._rebuild()

    # ---------- public API ----------
    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, query: str, namespace: str = "default") -> Optional[Match]:
        t0 = time.perf_counter()
        qv = self.embedder.embed(query)
        with self._lock:
            self._purge_expired()
            best: Optional[Match] = None
            if len(self._ids):
                scores = self._matrix @ qv  # cosine (vectors are normalised)
                for idx in np.argsort(-scores):
                    e = self._entries[self._ids[idx]]
                    if e.namespace != namespace:
                        continue
                    s = float(scores[idx])
                    if s >= self.threshold:
                        best = Match(e, s)
                    break  # first same-namespace candidate is the best one
            self.stats.lookup_seconds += time.perf_counter() - t0
            if best:
                best.entry.hits += 1
                self._entries.move_to_end(best.entry.id)
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return best

    def store(self, query: str, response: str, namespace: str = "default", ttl: Optional[float] = None) -> Entry:
        vec = self.embedder.embed(query)
        with self._lock:
            now = self._clock()
            life = ttl if ttl is not None else self.ttl
            entry = Entry(
                id=self._next_id, query=query, response=response, vector=vec,
                namespace=namespace, created_at=now,
                expires_at=(now + life) if life is not None else None,
            )
            self._next_id += 1
            self._entries[entry.id] = entry
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)  # evict least recently used
                self.stats.evictions += 1
            self._rebuild()
            return entry

    def get_or_compute(self, query: str, compute: Callable[[str], str], namespace: str = "default") -> tuple[str, bool, float]:
        """Return (response, was_cached, similarity). Calls `compute` only on a miss."""
        match = self.lookup(query, namespace)
        if match:
            return match.entry.response, True, match.score
        t0 = time.perf_counter()
        response = compute(query)
        _ = time.perf_counter() - t0
        self.store(query, response, namespace)
        return response, False, 0.0

    def record_saved(self, seconds: float) -> None:
        with self._lock:
            self.stats.saved_seconds += seconds

    def clear(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                n = len(self._entries)
                self._entries.clear()
            else:
                ids = [i for i, e in self._entries.items() if e.namespace == namespace]
                for i in ids:
                    del self._entries[i]
                n = len(ids)
            self._rebuild()
            return n

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        with self._lock:
            data = [
                {"query": e.query, "response": e.response, "namespace": e.namespace,
                 "created_at": e.created_at, "expires_at": e.expires_at, "hits": e.hits,
                 "vector": e.vector.tolist()}
                for e in self._entries.values()
            ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"threshold": self.threshold, "entries": data}, f)

    def load(self, path: str) -> int:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        with self._lock:
            for d in blob["entries"]:
                e = Entry(
                    id=self._next_id, query=d["query"], response=d["response"],
                    vector=np.asarray(d["vector"], dtype=np.float32), namespace=d["namespace"],
                    created_at=d["created_at"], expires_at=d["expires_at"], hits=d["hits"],
                )
                self._next_id += 1
                self._entries[e.id] = e
            self._purge_expired()
            self._rebuild()
            return len(self._entries)
