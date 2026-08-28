"""Query Qdrant for the top-k chunks most relevant to a question."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.config import get_settings
from app.embeddings import embed_query


@dataclass
class RetrievedChunk:
    text: str
    source: str
    heading_path: list[str]
    score: float


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)

    if not client.collection_exists(settings.qdrant_collection):
        return []

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=embed_query(question),
        limit=top_k or settings.top_k,
    ).points

    return [
        RetrievedChunk(
            text=point.payload["text"],
            source=point.payload["source"],
            heading_path=point.payload.get("heading_path", []),
            score=point.score,
        )
        for point in results
    ]
