"""Central configuration. Production rule #1: never hardcode."""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # --- Models (served by Ollama) ---
    EMBEDDING_MODEL: str = "nomic-embed-text"   # text -> 768-dim vectors
    LLM_MODEL: str = "gemma3:1b"                  # reasoning + JSON grading
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- Vector DB ---
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "insightdesk_docs"
    EMBED_DIM: int = 768                          # must match EMBEDDING_MODEL output

    # --- Retrieval knobs (tunable, not magic numbers) ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_DENSE: int = 6
    TOP_K_BM25: int = 6
    TOP_K_FINAL: int = 4
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # --- Agent ---
    MAX_RETRIES: int = 0                        # loop guard for CRAG corrections

    # --- API / Observability ---
    API_TOKEN: str = "dev"
    LANGCHAIN_TRACING_V2: bool = False

    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()
