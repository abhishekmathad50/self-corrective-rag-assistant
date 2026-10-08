from app.graph import GraphState


def rewrite_query(state: GraphState) -> GraphState:
    question = state["question"]
    retry_count = state["retry_count"]

    rewritten_query = f"{question} FastAPI documentation"

    return {
        **state,
        "rewritten_query": rewritten_query,
        "retry_count": retry_count + 1,
    }