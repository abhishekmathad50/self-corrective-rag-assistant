from app.graph import GraphState
from app.retrieval import retrieve_documents


def retrieval_node(state: GraphState) -> GraphState:
    query = state["rewritten_query"]

    documents = retrieve_documents(query, k=3)

    return {
        **state,
        "documents": documents,
    }