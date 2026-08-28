"""Local CPU embedding model, no GPU required.

E5 models expect a task prefix on the input text ("query: " for search
queries, "passage: " for indexed documents); this is part of how the model
was trained and measurably improves retrieval quality.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model, device="cpu")


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {text}" for text in texts]
    return _model().encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(f"query: {text}", normalize_embeddings=True).tolist()


def embedding_dim() -> int:
    return _model().get_embedding_dimension()
