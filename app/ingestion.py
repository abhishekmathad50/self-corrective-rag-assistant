
from pathlib import Path
import hashlib

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -----------------------------
# Configuration
# -----------------------------

DOCUMENTS_DIR = Path("data/documents")
CHROMA_DIR = Path("chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Embedding Model
# -----------------------------

def get_embeddings():
    """Initialize the embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# -----------------------------
# Vector Store
# -----------------------------

def get_vector_store():
    """Connect to the existing ChromaDB collection."""

    if not CHROMA_DIR.exists():
        raise ValueError("ChromaDB collection does not exist.")

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


# -----------------------------
# Load Documents
# -----------------------------

def load_documents():
    """Load Markdown and PDF documents."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    # Load Markdown files
    markdown_loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents.extend(markdown_loader.load())

    # Load PDF files
    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):

        try:
            pdf_loader = PyPDFLoader(str(pdf_path))
            pdf_documents = pdf_loader.load()

            documents.extend(pdf_documents)

            print(
                f"Loaded PDF: {pdf_path.name} "
                f"({len(pdf_documents)} pages)"
            )

        except Exception as error:
            print(
                f"Could not load {pdf_path.name}: {error}"
            )

    print(
        f"Total documents/pages loaded: {len(documents)}"
    )

    return documents


# -----------------------------
# Split Documents
# -----------------------------

def split_documents(documents):
    """Split documents into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# -----------------------------
# Generate Chunk IDs
# -----------------------------

def generate_chunk_ids(chunks):
    """Generate deterministic IDs for document chunks."""

    chunk_ids = []

    for chunk in chunks:

        source = str(
            chunk.metadata.get("source", "unknown")
        )

        page = str(
            chunk.metadata.get("page", "0")
        )

        start_index = str(
            chunk.metadata.get("start_index", "0")
        )

        unique_content = (
            f"{source}:{page}:{start_index}:{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            unique_content.encode("utf-8")
        ).hexdigest()

        chunk_ids.append(chunk_id)

    return chunk_ids


# -----------------------------
# Create Vector Store
# -----------------------------

def create_vector_store(chunks):
    """Rebuild ChromaDB without accumulating duplicate chunks."""

    if not chunks:
        raise ValueError(
            "No document chunks available for ingestion."
        )

    embeddings = get_embeddings()

    # Remove the previous collection before rebuilding
    if CHROMA_DIR.exists():

        try:
            existing_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
            )

            existing_store.delete_collection()

            print("Previous ChromaDB collection cleared.")

        except Exception as error:
            print(
                f"Could not clear previous collection: {error}"
            )
            raise

    chunk_ids = generate_chunk_ids(chunks)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids,
        persist_directory=str(CHROMA_DIR),
    )

    print(
        f"Successfully stored {len(chunks)} chunks in ChromaDB."
    )

    return vector_store


# -----------------------------
# Delete Document from ChromaDB
# -----------------------------

def delete_document_from_vector_store(filename):
    """Delete all vector chunks belonging to a document."""

    if not CHROMA_DIR.exists():
        print("ChromaDB does not exist. Nothing to delete.")
        return 0

    vector_store = get_vector_store()

    # Match the relative source path stored by document loaders
    source_path = str(DOCUMENTS_DIR / filename)

    collection = vector_store._collection

    # Retrieve matching chunk IDs
    results = collection.get(
        where={"source": source_path},
        include=[],
    )

    chunk_ids = results.get("ids", [])

    if not chunk_ids:
        print(
            f"No chunks found for {filename}. "
            "Document metadata may use a different path format."
        )
        return 0

    # Delete only matching document chunks
    collection.delete(ids=chunk_ids)

    print(
        f"Deleted {len(chunk_ids)} chunks "
        f"belonging to {filename}."
    )

    return len(chunk_ids)


# -----------------------------
# Complete Ingestion Pipeline
# -----------------------------

def ingest_documents():
    """Execute the complete ingestion pipeline."""

    documents = load_documents()

    if not documents:
        raise ValueError(
            f"No supported documents found in {DOCUMENTS_DIR}"
        )

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    print(
        "Document ingestion completed successfully."
    )

    return vector_store


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    ingest_documents()
