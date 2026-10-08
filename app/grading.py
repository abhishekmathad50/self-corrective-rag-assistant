from app.graph import GraphState
from app.llm import get_llm


def grade_documents(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]

    llm = get_llm()
    relevant_documents = []

    for document in documents:

        prompt = f"""
You are a relevance grader for a technical documentation RAG system.

Determine whether the document chunk contains information that could
help answer the user's question.

User question:
{question}

Document chunk:
{document.page_content}

If the document is relevant or related to the question, answer YES.
If it is clearly unrelated, answer NO.

Return ONLY one word:
YES
or
NO
"""

        response = llm.invoke(prompt)

        if isinstance(response.content, list):
            result = response.content[0]["text"].strip().upper()
        else:
            result = response.content.strip().upper()

        if result.startswith("YES"):
            relevant_documents.append(document)

    return {
        **state,
        "relevant_documents": relevant_documents,
    }