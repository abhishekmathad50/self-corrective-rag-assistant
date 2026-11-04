from app.graph import GraphState
from app.llm import get_llm


def query_analysis(state: GraphState) -> GraphState:
    """
    Convert the user's question into a standalone
    search query using conversation history.
    """

    question = state["question"]

    # Get previous conversation messages
    chat_history = state.get("chat_history", [])

    # Keep the most recent 6 messages
    recent_history = chat_history[-6:]

    history_text = "\n".join(
        f"{message.get('role', 'user').upper()}: "
        f"{message.get('content', '')}"
        for message in recent_history
    )

    llm = get_llm()

    prompt = f"""
You are a query rewriting assistant for a technical documentation
RAG system.

Your task is to convert the user's latest question into a
standalone search query suitable for retrieving information
from technical documentation.

Use conversation history only to understand follow-up questions.

Instructions:
1. Resolve references such as "it", "that", "this", and "they".
2. Use previous messages when the latest question depends on context.
3. Preserve the user's original intent.
4. Do not answer the question.
5. Do not introduce unrelated topics.
6. If the question is already standalone, keep its meaning unchanged.
7. Keep the rewritten query concise and specific.
8. Do not include explanations or additional text.
9. Return only the rewritten search query.

Conversation history:
{history_text if history_text else "No previous conversation."}

Latest user question:
{question}

Standalone search query:
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
            rewritten_query = question

    except Exception as error:
        print(f"Query analysis error: {error}")
        rewritten_query = question

    print(f"Original question: {question}")
    print(f"Rewritten query: {rewritten_query}")

    return {
        **state,
        "rewritten_query": rewritten_query,
    }
