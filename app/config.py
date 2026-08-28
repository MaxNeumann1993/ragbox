"""Application configuration, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM: any OpenAI-compatible endpoint (cloud API or local vLLM server)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Embeddings: runs locally on CPU, no GPU required
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "ragbox"

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4

    sample_docs_dir: str = "sample_docs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
