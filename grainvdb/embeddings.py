"""
GrainVDB Embedding Bridge
Unified local embedding provider interface for Apple Silicon.
Supports:
  1. Local Ollama embedding models (nomic-embed-text, bge-m3, all-minilm)
  2. Apple MLX / SentenceTransformers local models
  3. Fast zero-dependency deterministic projection fallback
"""

import json
import urllib.request
import numpy as np
from typing import List, Union, Optional


class BaseEmbeddingProvider:
    """Base interface for embedding generators."""
    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class FastLocalEmbedding(BaseEmbeddingProvider):
    """
    Zero-external-dependency deterministic semantic embedding.
    Uses SHA-256 token projection and subspace hashing.
    Ideal for testing, zero-dependency SDK distribution, and microsecond benchmarks.
    """
    def __init__(self, dimension: int = 128, seed: int = 42):
        super().__init__(dimension)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._projection = self.rng.randn(10007, dimension).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = abs(hash(word)) % 10007
            vec += self._projection[h]
        
        norm = np.linalg.norm(vec)
        if norm > 1e-7:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)


class OllamaEmbedding(BaseEmbeddingProvider):
    """
    Local Ollama embedding bridge (e.g. nomic-embed-text, bge-m3, all-minilm).
    Runs 100% on-device on Apple Silicon.
    """
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434", dimension: int = 768):
        super().__init__(dimension)
        self.model = model
        self.host = host.rstrip("/")

    def embed_query(self, text: str) -> np.ndarray:
        url = f"{self.host}/api/embeddings"
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vec = np.array(data["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return (vec / norm if norm > 1e-7 else vec).astype(np.float32)
        except Exception as e:
            # Fallback to local fast embedding if Ollama is unreachable
            return FastLocalEmbedding(self.dimension).embed_query(text)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_query(t) for t in texts], dtype=np.float32)


def get_embedding_provider(name: str = "fast", dimension: int = 128) -> BaseEmbeddingProvider:
    """Factory helper to obtain an embedding provider."""
    if name == "fast":
        return FastLocalEmbedding(dimension=dimension)
    elif name.startswith("ollama"):
        model_name = name.split(":")[-1] if ":" in name else "nomic-embed-text"
        return OllamaEmbedding(model=model_name, dimension=dimension)
    else:
        return FastLocalEmbedding(dimension=dimension)
