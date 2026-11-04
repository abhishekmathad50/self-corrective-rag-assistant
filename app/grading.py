
from app.graph import GraphState
from app.llm import get_llm


def grade_documents(state: GraphState) -> GraphState:
    """
    Evaluate retrieved document chunks for relevance
    to the user's original question.
    """

    question = state["question"]
    documents = state["documents"]

    # Handle empty retrieval results
    if not documents:
        return {
            **state,
            "relevant_documents": [],
        }

    llm = get_llm()
    relevant_documents = []

    for document in documents:

        prompt = f"""
You are a relevance grader for a technical documentation RAG system.

Your task is to determine whether the document chunk contains
information that can help answer the user's question.

USER QUESTION:
{question}

DOCUMENT CHUNK:
{document.page_content}

GRADING INSTRUCTIONS:
1. Consider technical terminology and related concepts.
2. Accept documents containing relevant supporting information.
3. Reject documents that are clearly unrelated.
4. Do not judge relevance based only on exact keyword matches.
5. Do not generate an explanation.

Return exactly one word:
YES
or
NO
"""

        try:
            response = llm.invoke(prompt)

            content = response.content

            if isinstance(content, list):
                result = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                ).strip().upper()
            else:
                result = str(content).strip().upper()

            # Accept only a clear YES response
            if result == "YES":
                relevant_documents.append(document)

        except Exception as error:
            print(
                f"Document grading error: {error}"
            )

            # Do not accept ungraded documents as relevant
            continue

    print(f"Documents received for grading: {len(documents)}")
    print(f"Relevant documents after grading: {len(relevant_documents)}")

    return {
        **state,
        "relevant_documents": relevant_documents,
    }
