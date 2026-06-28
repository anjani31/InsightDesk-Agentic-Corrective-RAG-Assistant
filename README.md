# 🏗️ InsightDesk — Agentic Corrective-RAG Assistant

⁠A production-grade, fully-local,Agentic ,Corrective RAG (CRAG) assistant that
	⁠grades its own retrieved documents, rewrites weak queries, falls back to web search,and verifies its own answers for hallucinations — built with LangGraph Qwen2.5 and nomic-embed-text (via Ollama), Qdrant, hybrid retrieval, and a
	⁠*cross-encoder reranker.

### Project Flow and Architecture
![alt text](<Project Architecture.jpeg>)


## Project Structure


insightdesk/
├── app/
│   ├── config.py          # central settings (no hardcoding)
│   ├── ingestion.py       # load → chunk → embed → store (offline)
│   ├── retrieval.py       # hybrid retriever + cross-encoder reranker
│   ├── graph_state.py     # shared State (the "baton")
│   ├── guardrails.py      # input/output guards + PII redaction
│   ├── nodes.py           # all LangGraph nodes (the agents)
│   ├── graph.py           # assemble nodes + edges → runnable agent
│   ├── api.py             # FastAPI gateway + streaming
│   └── evaluate.py        # RAGAS evaluation pipeline
├── tests/                 # pytest unit + smoke tests
├── data/                  # your source PDFs (gitignored)
├── docker-compose.yml     # Qdrant service
├── requirements.txt
├── Makefile
├── .env.example
└── README.md

##  Features

 ⁠🤖 Multi-agent LangGraph orchestration (router, retriever, graders, rewriter, generator)
 ⁠🔁 Corrective RAG (CRAG) — grades documents; rewrites query or falls back to web on poor retrieval
 ⁠🪞 Self-RAG groundedness check* — verifies answers against context before responding
 ⁠🔍 Hybrid retrieval* — dense (Qdrant) + sparse (BM25) fused, then cross-encoder reranking
 ⁠🛡️ Guardrails* — prompt-injection screening, PII redaction, output moderation
 ⁠🧠 Memory & recovery — LangGraph checkpointer (SQLite/Postgres)
 ⁠📊 Evaluation* — RAGAS (faithfulness, answer relevancy, context precision/recall)
 ⁠📹 Observability — LangSmith tracing
 ⁠🚪 FastAPI gateway — bearer auth + token-by-token *streaming + citations
 ⁠💸 100% free & local — no paid API keys required


## 🧰 Tech Stack
• Embeddings → Ollama · nomic‑embed‑text
• Free, runs locally
• 768‑dimensional vectors for semantic search
• LLM → Ollama · qwen2.5
• Strong reasoning ability
• Handles JSON grading and structured outputs
• Vector Database → Qdrant
• Supports filtering and hybrid search
• Docker‑native, easy deployment
• Sparse Retrieval → BM25 (rank‑bm25)
• Exact keyword and ID matching
• Complements dense embeddings
• Reranker → BAAI/bge‑reranker‑base
• Improves precision by re‑ordering candidates
• Cross‑encoder style reranking
• Orchestration → LangGraph
• Enables loops, branching, multi‑agent workflows
• Graph‑based orchestration for complex pipelines
• API → FastAPI
• Async support
• Streaming responses for efficiency
• Memory → LangGraph checkpointer
• Provides continuity across sessions
• Recovery mechanism for agent workflows
• Evaluation → RAGAS
• Measures faithfulness of answers
• Evaluates context recall in RAG pipelines
• Observability → LangSmith
• Tracing for debugging
• Tracks cost and latency for optimization


## License

MIT ©️ 2026 Anjani Singh