from app.graph import GraphState
from app.retrieval import retrieve_documents


def retrieval_node(state: GraphState) -> GraphState:
    query = state["rewritten_query"]
    selected_documents = state.get("selected_documents", [])

    documents = retrieve_documents(
        query=query,
        k=3,
        selected_documents=selected_documents or None,
    )

    return {
        **state,
        "documents": documents,
    }