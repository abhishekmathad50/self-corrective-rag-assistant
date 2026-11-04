
from app.graph import GraphState
from app.llm import get_llm


def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrite the original question to improve document retrieval.
    """

    question = state["question"]
    previous_query = state.get("rewritten_query", "")
    retry_count = state["retry_count"]

    llm = get_llm()

    prompt = f"""
You are a search query optimization assistant
for a technical documentation RAG system.

The previous retrieval attempt did not find relevant documents.

Original user question:
{question}

Previous search query:
{previous_query or question}

Instructions:
1. Preserve the original question's intent.
2. Use clearer technical terminology.
3. Include relevant technical keywords.
4. Remove unnecessary words.
5. Do not answer the question.
6. Do not introduce unrelated topics.
7. Return only the improved search query.
"""

    try:
        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):
            rewritten_query = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            ).strip()
        else:
            rewritten_query = str(content).strip()

        if not rewritten_query:
            rewritten_query = previous_query or question

    except Exception as error:
        print(f"Query rewriting error: {error}")
        rewritten_query = previous_query or question

    print(f"Original question: {question}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Retry count: {retry_count + 1}")

    return {
        **state,
        "rewritten_query": rewritten_query,
        "retry_count": retry_count + 1,
    }
