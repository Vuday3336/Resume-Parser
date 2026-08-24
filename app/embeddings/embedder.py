"""Wraps OpenAI embeddings with simple batching. Embeddings are the backbone of stage-1
matching (see app/matching/vector_match.py) — cheap enough to run on every resume/JD, unlike
the LLM rerank which we deliberately restrict to a shortlist."""
import numpy as np
from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings.require_openai()
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed_resume_text(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    client = _get_client()
    # OpenAI silently truncates at the model's context window; guard against empty strings,
    # which the API rejects outright.
    safe_texts = [t if t.strip() else " " for t in texts]
    response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=safe_texts)
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
