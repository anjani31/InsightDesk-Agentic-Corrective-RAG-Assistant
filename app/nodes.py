"""All LangGraph nodes (the multi-agent workforce)."""
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel, Field
from app.config import settings
from app.graph_state import RAGState
from app.guardrails import input_guard, output_guard, redact_pii
from langchain_core.messages import HumanMessage, AIMessage

# Shared deterministic brain
llm = ChatOllama(
    model=settings.LLM_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0,
    # Add timeout to prevent hangs
    request_timeout=60,
)

RETRIEVER = None


# ============================================================
# GUARD INPUT
# ============================================================
def guard_input(state: RAGState) -> dict:
    safe, reason = input_guard(state["question"])
    if not safe:
        return {
            "generation": f"I can't process that request. ({reason})",
            "documents": [],
            "datasource": "blocked",
            "retry_count": 0,
        }
    return {
        "question": redact_pii(state["question"]),
        "retry_count": 0,
    }


# ============================================================
# ROUTER
# ============================================================
class RouteDecision(BaseModel):
    datasource: str = Field(description="'vectorstore' or 'web'")


def route_question(state: RAGState) -> dict:
    try:
        router = llm.with_structured_output(RouteDecision)
        decision = router.invoke(
            "Route the question. If it concerns our internal product, policies or docs, "
            "answer 'vectorstore'. If it's general/current-events, answer 'web'.\n\n"
            f"Question: {state['question']}"
        )
        ds = decision.datasource
        # Validate
        if ds not in ("vectorstore", "web"):
            ds = "vectorstore"  # safe default
    except Exception as e:
        print(f"[Router] Failed: {e}, defaulting to vectorstore")
        ds = "vectorstore"
    return {"datasource": ds}


def route_edge(state: RAGState) -> str:
    return "retrieve" if state["datasource"] == "vectorstore" else "web_search"


# ============================================================
# RETRIEVE
# ============================================================
def retrieve(state: RAGState) -> dict:
    query = state.get("rewritten_question") or state["question"]
    try:
        docs = RETRIEVER.retrieve(query)
    except Exception as e:
        print(f"[Retrieve] Failed: {e}")
        docs = []
    return {"documents": docs}


# ============================================================
# GRADE DOCUMENTS - with safety net
# ============================================================
class GradeDoc(BaseModel):
    relevant: str = Field(description="'yes' if helpful, else 'no'")


def grade_documents(state: RAGState) -> dict:
    """Grade documents. If grader fails, KEEP all docs (don't drop them)."""
    if not state.get("documents"):
        return {"documents": []}

    grader = llm.with_structured_output(GradeDoc)
    kept = []
    failed_count = 0

    for d in state["documents"]:
        try:
            res = grader.invoke(
                f"Question: {state['question']}\n\n"
                f"Document: {d.page_content[:500]}\n\n"
                "Is this document relevant? Answer yes or no."
            )
            if res.relevant.lower().strip() == "yes":
                kept.append(d)
        except Exception as e:
            # If grader fails, KEEP the doc (don't lose info)
            print(f"[Grader] Failed for doc, keeping it: {e}")
            kept.append(d)
            failed_count += 1

    # Safety net: if we dropped everything due to grader failures,
    # keep original docs
    if not kept and state["documents"]:
        print("[Grader] All docs filtered out, keeping originals")
        kept = state["documents"]

    return {"documents": kept}


def decide_after_grading(state: RAGState) -> str:
    """Hard limits to prevent infinite loops."""
    retry_count = state.get("retry_count", 0)

    # ABSOLUTE HARD STOP
    if retry_count >= settings.MAX_RETRIES:
        print(f"[Decision] Max retries reached ({retry_count}/{settings.MAX_RETRIES})")
        if state["documents"]:
            return "generate"
        return "web_search"  # Final fallback

    # If no docs, try to rewrite
    if len(state["documents"]) == 0:
        return "rewrite"

    # We have docs, proceed
    return "generate"


# ============================================================
# REWRITE QUERY
# ============================================================
def rewrite_query(state: RAGState) -> dict:
    """Rewrite query for better retrieval. Always increments retry counter."""
    current = state.get("rewritten_question") or state["question"]

    try:
        chain = (
            ChatPromptTemplate.from_template(
                "The search returned poor results. Rewrite this question to be clearer "
                "and richer in keywords for retrieval.\n\n"
                "Original: {q}\n\n"
                "Rewritten:"
            )
            | llm
            | StrOutputParser()
        )
        better = chain.invoke({"q": current})
    except Exception as e:
        print(f"[Rewrite] Failed: {e}")
        better = current  # Use original if rewrite fails

    return {
        "rewritten_question": better,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ============================================================
# WEB SEARCH FALLBACK
# ============================================================
def web_search(state: RAGState) -> dict:
    try:
        results = DuckDuckGoSearchResults(num_results=4).invoke(state["question"])
        return {
            "documents": [
                Document(page_content=results, metadata={"source": "web"})
            ]
        }
    except Exception as e:
        print(f"[WebSearch] Failed: {e}")
        return {
            "documents": [
                Document(
                    page_content="No information found.",
                    metadata={"source": "none"},
                )
            ]
        }


# ============================================================
# GENERATE
# ============================================================
# ============================================================
# GENERATE
# ============================================================
def generate(state: RAGState) -> dict:
    """Generate answer with conversation memory."""

    # Build context from documents
    if state.get("documents"):
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content[:800]}"
            for d in state["documents"][:3]
        )
    else:
        context = "No documents retrieved."

    # Build conversation history
    history_messages = state.get("messages", [])
    history_text = ""
    for msg in history_messages[-6:]:
        if hasattr(msg, 'content'):
            role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

    # Prompt with strong memory emphasis
    chain = (
        ChatPromptTemplate.from_template(
            "You are InsightDesk with MEMORY of past conversations.\n\n"
            "===== PREVIOUS CONVERSATION =====\n"
            "{history}\n"
            "===== END =====\n\n"
            "Document context:\n{context}\n\n"
            "Current question: {question}\n\n"
            "CRITICAL RULES:\n"
            "1. You DO have memory of this conversation\n"
            "2. If user mentions their name or facts, recall from history\n"
            "3. Answer based on context AND history\n"
            "4. Be specific and helpful\n\n"
            "Answer:"
        )
        | llm
        | StrOutputParser()
    )

    try:
        answer = chain.invoke({
            "history": history_text or "(no prior conversation)",
            "context": context,
            "question": state["question"]
        })
    except Exception as e:
        print(f"[Generate] Failed: {e}")
        answer = "Error generating response."

    # Return messages for memory
    return {
        "generation": answer,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ]
    }



# ============================================================
# HALLUCINATION GRADER - Self-RAG
# ============================================================
class Grounded(BaseModel):
    grounded: str = Field(description="'yes' if supported by docs, else 'no'")


def grade_generation(state: RAGState) -> str:
    """Decide: use this answer, regenerate, or accept anyway."""

    # Hard stop: don't loop on grading
    if state.get("retry_count", 0) >= settings.MAX_RETRIES:
        return "useful"

    # Skip grading if no docs to check against
    if not state.get("documents"):
        return "useful"

    try:
        grader = llm.with_structured_output(Grounded)
        context = "\n\n".join(d.page_content[:300] for d in state["documents"])
        res = grader.invoke(
            f"Documents:\n{context}\n\n"
            f"Answer:\n{state['generation']}\n\n"
            "Is the answer supported by the documents? yes or no."
        )

        if res.grounded.lower().strip() == "yes":
            return "useful"

        # Not grounded, but stop trying after max retries
        if state["retry_count"] >= settings.MAX_RETRIES:
            return "useful"  # Accept whatever we have
        return "not_grounded"

    except Exception as e:
        # Grader failed -> accept the answer
        print(f"[HallucinationGrader] Failed: {e}")
        return "useful"


# ============================================================
# GUARD OUTPUT
# ============================================================
def guard_output(state: RAGState) -> dict:
    if state.get("datasource") == "blocked":
        return {}
    safe, clean = output_guard(state["generation"])
    return {"generation": clean if safe else "I'm unable to share that response."}
