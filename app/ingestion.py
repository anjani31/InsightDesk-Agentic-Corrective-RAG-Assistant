"""Offline pipeline: load -> chunk -> embed -> store. Run ONCE before serving."""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config import settings


def load_chunks(data_dir: str = "data"):
    """Load all PDFs and split them into overlapping, model-friendly chunks."""
    docs = []
    for pdf in Path(data_dir).glob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())   # page -> Document(text, metadata)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # try paragraphs first, then sentences, then words (keeps ideas whole):
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks


def build_index(data_dir: str = "data"):
    chunks = load_chunks(data_dir)

    embeddings = OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
    )

    client = QdrantClient(url=settings.QDRANT_URL)
    if not client.collection_exists(settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            # size MUST match the embedding model; COSINE compares meaning by angle:
            vectors_config=VectorParams(size=settings.EMBED_DIM, distance=Distance.COSINE),
        )

    store = QdrantVectorStore(
        client=client, collection_name=settings.COLLECTION_NAME, embedding=embeddings
    )
    store.add_documents(chunks)   # embeds each chunk via Ollama, then upserts
    print("Index built and stored in Qdrant")
    return [d.page_content for d in chunks]
                
            
            


if __name__ == "__main__":
    build_index()
