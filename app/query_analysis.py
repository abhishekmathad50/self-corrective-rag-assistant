
from app.graph import GraphState
from app.llm import get_llm


def query_analysis(state: GraphState) -> GraphState:
    question = state["question"]

    # Get previous conversation messages
    chat_history = state.get("chat_history", [])

    # Keep the most recent 6 messages
    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:
        role = message.get("role", "user").upper()
        content = message.get("content", "")

        history_text += f"{role}: {content}\n"

    llm = get_llm()

    prompt = f"""
You are a query rewriting assistant for a technical documentation
RAG system.

Your task is to convert the user's latest question into a
standalone search query suitable for retrieving information
from technical documentation.

Use the conversation history to understand follow-up questions.

Instructions:
1. Resolve references such as "it", "that", "this", and "they".
2. Use previous messages when the latest question depends on context.
3. Preserve the user's original intent.
4. Do not answer the question.
5. Do not introduce unrelated topics.
6. If the question is already standalone, keep its meaning unchanged.
7. Return only the rewritten search query.

Conversation history:
{history_text if history_text else "No previous conversation."}

Latest user question:
{question}

Standalone search query:
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        rewritten_query = " ".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
        ).strip()
    else:
        rewritten_query = str(response.content).strip()

    if not rewritten_query:
        rewritten_query = question

    print(f"Original question: {question}")
    print(f"Rewritten query: {rewritten_query}")

    return {
        **state,
        "rewritten_query": rewritten_query,
    }
