
from langgraph.graph import StateGraph, END

from app.graph import GraphState
from app.query_analysis import query_analysis
from app.retrieval_node import retrieval_node
from app.grading import grade_documents
from app.conditions import decide_next_step, no_answer
from app.rewrite import rewrite_query
from app.generation import generate_answer


def build_graph():
    """
    Build and compile the self-corrective RAG workflow.
    """

    workflow = StateGraph(GraphState)

    # Add workflow nodes
    workflow.add_node("query_analysis", query_analysis)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grading", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generation", generate_answer)
    workflow.add_node("no_answer", no_answer)

    # Set starting point
    workflow.set_entry_point("query_analysis")

    # Main retrieval and grading flow
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "grading")

    # Conditional routing based on document relevance
    workflow.add_conditional_edges(
        "grading",
        decide_next_step,
        {
            "generate": "generation",
            "rewrite": "rewrite",
            "no_answer": "no_answer",
        },
    )

    # Retry retrieval with an improved query
    workflow.add_edge("rewrite", "retrieval")

    # End workflow after answer generation or fallback
    workflow.add_edge("generation", END)
    workflow.add_edge("no_answer", END)

    return workflow.compile()


# -----------------------------
# Local Workflow Testing
# -----------------------------

if __name__ == "__main__":

    graph = build_graph()

    initial_state = {
        "question": "How do I define a path parameter in FastAPI?",
        "rewritten_query": "",
        "documents": [],
        "relevant_documents": [],
        "answer": "",
        "retry_count": 0,
        "sources": [],
        "selected_documents": [],
        "chat_history": [],
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print("-", source)

    print("\nRetry count:", result["retry_count"])
