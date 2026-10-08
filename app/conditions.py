from app.graph import GraphState


def decide_next_step(state: GraphState) -> str:
    relevant_documents = state["relevant_documents"]
    retry_count = state["retry_count"]

    if relevant_documents:
        return "generate"

    if retry_count < 2:
        return "rewrite"

    return "no_answer"


def no_answer(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": (
            "I couldn't find relevant information in the provided "
            "documentation to answer this question."
        ),
        "sources": [],
    }