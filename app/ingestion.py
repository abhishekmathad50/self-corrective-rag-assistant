from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
DOCUMENTS_DIR = Path("data/documents")
CHROMA_DIR = Path("chroma_db")


def load_documents():
    """Load all Markdown documents from the documents directory."""

    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    return documents


def split_documents(documents):
    """Split documents into smaller chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vector_store(chunks):
    """Create embeddings and store them in ChromaDB."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print("Documents stored in ChromaDB.")

    return vector_store


def ingest_documents():
    """Run the complete ingestion pipeline."""

    documents = load_documents()

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    return vector_store


if __name__ == "__main__":
    ingest_documents()