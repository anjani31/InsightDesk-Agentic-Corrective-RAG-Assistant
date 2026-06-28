"""The shared State (the "baton") passed between all nodes."""
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    question: str                 # the user's current question
    rewritten_question: str       # improved query (set by rewrite node)
    documents: List[Document]     # docs currently in play
    generation: str               # the final answer
    retry_count: int              # loop guard — prevents infinite correction loops
    datasource: str               # "vectorstore" | "web" | "blocked"
    # append (don't overwrite) chat history — that's the reducer:
    messages: Annotated[List[BaseMessage], add_messages]
