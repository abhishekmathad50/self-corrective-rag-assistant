
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ChromaDB directory
CHROMA_DIR = BASE_DIR / "chroma_db"

# Documents directory
DOCUMENTS_DIR = Path("data") / "documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum relevance score required for retrieval
SIMILARITY_THRESHOLD = 0.30


def get_vector_store():
    """Connect to the existing ChromaDB vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vector_store


def retrieve_documents(
    query: str,
    k: int = 3,
    selected_documents: list[str] | None = None,
):
    """
    Retrieve relevant chunks using a minimum similarity score.
    Optionally restrict retrieval to selected documents.
    """

    vector_store = get_vector_store()

    search_filter = None

    if selected_documents:
        selected_sources = [
            str(DOCUMENTS_DIR / name)
            for name in selected_documents
        ]

        search_filter = {
            "source": {
                "$in": selected_sources
            }
        }

    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=k,
        filter=search_filter,
    )

    # Keep only chunks meeting the minimum threshold
    relevant_documents = [
        document
        for document, score in results
        if score >= SIMILARITY_THRESHOLD
    ]

    print(f"Retrieved chunks: {len(results)}")
    print(f"Relevant chunks: {len(relevant_documents)}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    return relevant_documents


if __name__ == "__main__":

    query = "How do I define a path parameter in FastAPI?"

    documents = retrieve_documents(query)

    print(f"\nQuery: {query}")
    print(f"Final retrieved documents: {len(documents)}\n")

    for index, document in enumerate(documents, start=1):
        print("=" * 80)
        print(f"Result {index}")
        print(f"Source: {document.metadata.get('source')}")
        print(document.page_content[:500])
        print()
