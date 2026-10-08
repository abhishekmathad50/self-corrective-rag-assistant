from app.graph import GraphState
from app.llm import get_llm


def query_analysis(state: GraphState) -> GraphState:
    question = state["question"]

    llm = get_llm()

    prompt = f"""
Rewrite the following user question into a concise search query
that will work well for retrieving information from FastAPI technical
documentation.

Keep the meaning unchanged.
Return only the rewritten query.

User question:
{question}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        rewritten_query = response.content[0]["text"].strip()
    else:
        rewritten_query = response.content.strip()
    return {
        **state,
        "rewritten_query": rewritten_query,
    }