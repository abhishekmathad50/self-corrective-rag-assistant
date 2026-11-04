
import re
from pathlib import Path

from app.graph import GraphState
from app.llm import get_llm


def format_source(document):
    """Format a document source with its page number when available."""

    source = Path(
        document.metadata.get("source", "Unknown")
    ).name

    page = document.metadata.get("page")

    if page is not None:
        return f"{source}, page {int(page) + 1}"

    return source


def generate_answer(state: GraphState) -> GraphState:
    """Generate a grounded answer with accurate source citations."""

    question = state["question"]
    documents = state["relevant_documents"]
    chat_history = state.get("chat_history", [])

    # Keep the latest 6 conversation messages
    recent_history = chat_history[-6:]

    history_text = "\n".join(
        f"{message.get('role', 'user').upper()}: "
        f"{message.get('content', '')}"
        for message in recent_history
    )

    # Handle missing documents
    if not documents:
        return {
            **state,
            "answer": (
                "I don't have enough information in the provided "
                "documentation to answer this question."
            ),
            "sources": [],
        }

    # Assign one citation number to each unique source
    source_to_id = {}
    sources = []
    context_parts = []

    for document in documents:

        source = format_source(document)

        if source not in source_to_id:
            citation_id = len(sources) + 1

            source_to_id[source] = citation_id
            sources.append(source)

        citation_id = source_to_id[source]

        context_parts.append(
            f"[{citation_id}] SOURCE: {source}\n"
            f"CONTENT: {document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Generate answer
    llm = get_llm()

    valid_citation_ids = list(range(1, len(sources) + 1))

    prompt = f"""
You are a reliable technical documentation assistant.

Answer the latest user question using ONLY the retrieved
documentation as factual evidence.

CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation."}

LATEST USER QUESTION:
{question}

RETRIEVED DOCUMENTATION:
{context}

STRICT CITATION RULES:

1. Use only retrieved documentation for factual claims.
2. Cite every factual claim using the citation IDs shown in the context.
3. Valid citation IDs are: {valid_citation_ids}
4. Never invent or modify citation IDs.
5. Use the same citation ID when multiple chunks belong to the same source.
6. Cite only sources that directly support the claim.
7. Do not cite sources that do not support the claim.
8. If documentation is insufficient, clearly state that.
9. Use conversation history only to understand follow-up questions.
10. Do not treat previous assistant answers as verified facts.
11. Answer only the latest question.
12. Keep the response clear and technically accurate.
13. Keep answers concise and focused.
14. For simple questions, answer in 2–4 sentences.
15. Use bullet points for multi-step technical explanations.
16. Avoid repeating information.
17. Include only details necessary to answer the question.

Return only the answer with relevant numbered citations.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        answer = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()
    else:
        answer = str(content).strip()

    # Validate citation IDs against the displayed source list
    valid_citations = set(valid_citation_ids)

    def validate_citation(match):
        citation_number = int(match.group(1))

        if citation_number in valid_citations:
            return match.group(0)

        return ""

    answer = re.sub(
        r"\[(\d+)\]",
        validate_citation,
        answer,
    )

    # Clean up formatting
    answer = re.sub(r" {2,}", " ", answer)
    answer = re.sub(r" +([.,;:])", r"\1", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return {
        **state,
        "answer": answer.strip(),
        "sources": sources,
    }
