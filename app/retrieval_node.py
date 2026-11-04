from app.graph import GraphState
from app.retrieval import retrieve_documents


def retrieval_node(state: GraphState) -> GraphState:
    """
    Retrieve relevant document chunks using the original
    or rewritten query and optional document selection.
    """

    # Use rewritten query when available
    query = (
        state.get("rewritten_query")
        or state["question"]
    )

    # Get selected documents
    selected_documents = state.get("selected_documents", [])

    # Retrieve relevant document chunks
    documents = retrieve_documents(
        query=query,
        selected_documents=selected_documents or None,
    )

    return {
        **state,
        "documents": documents,
    }
