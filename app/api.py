"""FastAPI gateway: auth + streaming + memory."""

import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.config import settings
from app.graph import build_app
from app.ingestion import load_chunks

os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGCHAIN_TRACING_V2).lower()

app = FastAPI(title="InsightDesk RAG API")

# Build heavy objects ONCE at startup, never per request:
try:
    _CHUNKS = load_chunks("data")
except Exception:
    _CHUNKS = []
AGENT = build_app(all_chunks=_CHUNKS)


class Query(BaseModel):
    question: str
    thread_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Query, authorization: str = Header(None)):
    if authorization != f"Bearer {settings.API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    config = {"configurable": {"thread_id": body.thread_id}}

    def stream():
        for chunk in AGENT.stream(
            {"question": body.question, "retry_count": 0},
            config=config, stream_mode="updates",
        ):
            for _, update in chunk.items():
                if isinstance(update, dict) and update.get("generation"):
                    yield update["generation"]

    return StreamingResponse(stream(), media_type="text/plain")
