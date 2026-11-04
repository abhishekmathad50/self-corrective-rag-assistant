
from app.graph import GraphState


# Maximum number of query rewriting attempts
MAX_RETRIES = 2


def decide_next_step(state: GraphState) -> str:
    """
    Decide the next workflow step based on document relevance
    and the number of retrieval retries.
    """

    relevant_documents = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)

    # Generate an answer when relevant documents are available
    if relevant_documents:
        print("Relevant documents found. Proceeding to generation.")
        return "generate"

    # Retry retrieval when no relevant documents are found
    if retry_count < MAX_RETRIES:
        print(
            f"No relevant documents found. "
            f"Retrying retrieval ({retry_count + 1}/{MAX_RETRIES})."
        )
        return "rewrite"

    # Stop after maximum retries
    print("Maximum retries reached. Returning fallback response.")

    return "no_answer"


def no_answer(state: GraphState) -> GraphState:
    """
    Return a fallback response when relevant documents
    cannot be found after the allowed retries.
    """

    return {
        **state,
        "answer": (
            "I couldn't find relevant information in the provided "
            "documentation to answer this question. "
            "Please try rephrasing your question or selecting "
            "a different document."
        ),
        "sources": [],
    }
