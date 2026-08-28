"""Orchestrates the RAG loop: retrieve chunks -> build context -> ask the LLM -> attach sources."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.llm.base import LLMProvider
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.retriever import RetrievedChunk, retrieve

NO_CONTEXT_ANSWER = (
    "I don't have any ingested documents to search yet. Run the ingest step "
    "first, then ask again."
)


@dataclass
class Answer:
    text: str
    sources: list[str]


def _build_provider() -> LLMProvider:
    settings = get_settings()
    return OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        heading = " > ".join(chunk.heading_path)
        label = f"[{chunk.source}{f': {heading}' if heading else ''}]"
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, provider: LLMProvider | None = None) -> Answer:
    chunks = retrieve(question)
    if not chunks:
        return Answer(text=NO_CONTEXT_ANSWER, sources=[])

    context = _build_context(chunks)
    provider = provider or _build_provider()
    answer_text = provider.answer(question, context)

    sources = sorted({chunk.source for chunk in chunks})
    return Answer(text=answer_text, sources=sources)
