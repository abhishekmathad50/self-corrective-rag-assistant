from typing import TypedDict, NotRequired


class GraphState(TypedDict):
    """State definition for the RAG workflow."""

    question: str
    rewritten_query: NotRequired[str]
    documents: list
    relevant_documents: list
    answer: NotRequired[str]
    retry_count: int
    sources: NotRequired[list]
    selected_documents: NotRequired[list[str]]
    chat_history: NotRequired[list[dict]]
