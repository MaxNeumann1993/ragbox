"""Load documents from a folder, chunk them, embed them, and store them in Qdrant."""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import embed_passages, embedding_dim

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def _get_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def _ensure_collection(client: QdrantClient) -> None:
    settings = get_settings()
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=embedding_dim(), distance=Distance.COSINE),
    )


def ingest_directory(directory: str | Path) -> int:
    """Ingest every supported file in `directory` into Qdrant. Returns the number of chunks stored."""
    settings = get_settings()
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        return 0

    client = _get_client()
    _ensure_collection(client)

    total_chunks = 0
    for path in files:
        text = _read_file(path)
        chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        if not chunks:
            continue

        vectors = embed_passages([c.content for c in chunks])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "heading_path": chunk.heading_path,
                    "source": path.name,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total_chunks += len(points)

    return total_chunks
