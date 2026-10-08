from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# ChromaDB directory
CHROMA_DIR = BASE_DIR / "chroma_db"


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vector_store


def retrieve_documents(query: str, k: int = 3):
    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query,
        k=k
    )

    return documents


if __name__ == "__main__":

    query = "How do I define a path parameter in FastAPI?"

    documents = retrieve_documents(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(documents)} chunks\n")

    for i, document in enumerate(documents, start=1):

        print("=" * 80)
        print(f"Result {i}")

        print(
            f"Source: "
            f"{document.metadata.get('source')}"
        )

        print(
            document.page_content[:500]
        )

        print()