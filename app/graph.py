from typing import TypedDict


class GraphState(TypedDict):
    question: str
    rewritten_query: str
    documents: list
    relevant_documents: list
    answer: str
    retry_count: int
    sources: list