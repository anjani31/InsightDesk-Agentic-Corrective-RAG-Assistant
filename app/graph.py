"""Assemble all nodes + edges into one runnable agent."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.graph_state import RAGState
from app import nodes
from app.retrieval import HybridRetriever


def build_app(all_chunks):
    nodes.RETRIEVER = HybridRetriever(all_chunks)

    g = StateGraph(RAGState)

    # Add nodes
    g.add_node("guard_input", nodes.guard_input)
    g.add_node("router", nodes.route_question)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite", nodes.rewrite_query)
    g.add_node("web_search", nodes.web_search)
    g.add_node("generate", nodes.generate)
    g.add_node("guard_output", nodes.guard_output)

    # Edges
    g.add_edge(START, "guard_input")

    g.add_conditional_edges(
        "guard_input",
        lambda s: "guard_output" if s.get("datasource") == "blocked" else "router",
        {"guard_output": "guard_output", "router": "router"},
    )

    g.add_conditional_edges(
        "router", nodes.route_edge,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )

    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges(
        "grade_documents", nodes.decide_after_grading,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )

    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", "guard_output")
    g.add_edge("guard_output", END)

    memory = InMemorySaver()
    return g.compile(checkpointer=memory)
