"""Hybrid retrieval (dense + BM25) + cross-encoder reranking."""
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from huggingface_hub import hf_hub_download


from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from app.config import settings


def _dense_retriever():
    """Semantic retriever backed by Qdrant + Ollama embeddings."""
    embeddings = OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
    )
    store = QdrantVectorStore(
        client=QdrantClient(url=settings.QDRANT_URL),
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings,
    )
    return store.as_retriever(search_kwargs={"k": settings.TOP_K_DENSE})


def _bm25_retriever(all_chunks):
    """Keyword retriever — catches exact tokens (IDs, codes) embeddings miss."""
    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = settings.TOP_K_BM25
    return bm25


class HybridRetriever:
    """Wide net (hybrid) -> precise filter (rerank)."""

    def __init__(self, all_chunks):
        # Load the cross-encoder once and reuse (it reads query+doc TOGETHER):
        self._reranker = CrossEncoder(settings.RERANKER_MODEL)
        # EnsembleRetriever fuses both rankings via Reciprocal Rank Fusion:
        self._ensemble = EnsembleRetriever(
            retrievers=[_dense_retriever(), _bm25_retriever(all_chunks)],
            weights=[0.6, 0.4],   # 60% semantic, 40% keyword (tunable)
        )

    def retrieve(self, query: str):
        candidates = self._ensemble.invoke(query)            # wide candidate set
        if not candidates:
            return []
        pairs = [(query, d.page_content) for d in candidates]
        scores = self._reranker.predict(pairs)               # precise relevance scores
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: settings.TOP_K_FINAL]]
