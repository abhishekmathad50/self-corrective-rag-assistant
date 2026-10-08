from app.graph import GraphState
from app.llm import get_llm


def generate_answer(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["relevant_documents"]

    if not documents:
        return {
            **state,
            "answer": "I don't have enough information in the provided documentation to answer this question.",
            "sources": [],
        }

    context_parts = []

    for i, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    llm = get_llm()

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question using ONLY the provided documentation.

If the documentation does not contain enough information, say:
"I don't have enough information in the provided documentation."

Do not invent information.

User question:
{question}

Documentation:
{context}

Give a concise and clear answer.
Include the relevant source filename at the end.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        answer = response.content[0]["text"].strip()
    else:
        answer = response.content.strip()

    sources = list(
        dict.fromkeys(
            document.metadata.get("source", "Unknown")
            for document in documents
        )
    )

    return {
        **state,
        "answer": answer,
        "sources": sources,
    }