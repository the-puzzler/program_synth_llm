from __future__ import annotations

import math
import os
from typing import Sequence

from openai import OpenAI


def _openrouter_client() -> OpenAI:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for embeddings.")
    return OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )


def embed_texts(texts: Sequence[str], *, model: str = "qwen/qwen3-embedding-8b") -> list[list[float]]:
    if not texts:
        return []
    client = _openrouter_client()
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length for cosine similarity.")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(vec_a, vec_b):
        af = float(a)
        bf = float(b)
        dot += af * bf
        na += af * af
        nb += bf * bf
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)

