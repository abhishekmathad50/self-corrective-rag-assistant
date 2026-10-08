from langgraph.graph import StateGraph, END

from app.graph import GraphState
from app.query_analysis import query_analysis
from app.retrieval_node import retrieval_node
from app.grading import grade_documents
from app.conditions import decide_next_step, no_answer
from app.rewrite import rewrite_query
from app.generation import generate_answer


def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("query_analysis", query_analysis)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grading", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generation", generate_answer)
    workflow.add_node("no_answer", no_answer)

    # Starting point
    workflow.set_entry_point("query_analysis")

    # Main flow
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "grading")

    # Conditional routing
    workflow.add_conditional_edges(
        "grading",
        decide_next_step,
        {
            "generate": "generation",
            "rewrite": "rewrite",
            "no_answer": "no_answer",
        },
    )

    # Retry loop
    workflow.add_edge("rewrite", "retrieval")

    # Final paths
    workflow.add_edge("generation", END)
    workflow.add_edge("no_answer", END)

    return workflow.compile()


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
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print("-", source)