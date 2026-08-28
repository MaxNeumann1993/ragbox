"""FastAPI application exposing the RAG loop as an HTTP API."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import get_settings
from app.ingest import ingest_directory
from app.rag import ask

app = FastAPI(
    title="ragbox",
    description="A small, self-hostable RAG system for your own documents.",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestRequest(BaseModel):
    directory: str | None = None


class IngestResponse(BaseModel):
    chunks_ingested: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    directory = request.directory or get_settings().sample_docs_dir
    if not Path(directory).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    try:
        count = ingest_directory(directory)
    except ResponseHandlingException as exc:
        raise HTTPException(status_code=503, detail=f"Could not reach Qdrant: {exc}") from exc

    return IngestResponse(chunks_ingested=count)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = ask(request.question)
    except ResponseHandlingException as exc:
        raise HTTPException(status_code=503, detail=f"Could not reach Qdrant: {exc}") from exc

    return AskResponse(answer=result.text, sources=result.sources)


static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")
