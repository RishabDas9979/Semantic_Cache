"""Pluggable text embedders.

HashingEmbedder is dependency-free (numpy only) and works offline: it combines
word unigrams/bigrams and character n-grams hashed into a fixed-size vector.
SentenceTransformerEmbedder gives true semantic similarity when the optional
`sentence-transformers` package (and model weights) are available.
"""
from __future__ import annotations

import re
import zlib
from abc import ABC, abstractmethod

import numpy as np

_STOPWORDS = frozenset(
    "a an the is are was were be been am do does did to of in on for with and or "
    "i you we it this that my me your can could would should please how what".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised float32 vector of shape (dim,)."""


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512, char_ngram: int = 3):
        self.dim = dim
        self.char_ngram = char_ngram

    def _add(self, vec: np.ndarray, feature: str, weight: float) -> None:
        h = zlib.crc32(feature.encode("utf-8"))
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[h % self.dim] += sign * weight

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        content = [t for t in tokens if t not in _STOPWORDS] or tokens
        for t in content:
            self._add(vec, "w:" + t, 1.0)
            padded = f"#{t}#"
            n = self.char_ngram
            for i in range(max(1, len(padded) - n + 1)):
                self._add(vec, "c:" + padded[i : i + n], 0.3)
        for a, b in zip(content, content[1:]):
            self._add(vec, f"b:{a}_{b}", 0.5)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)
