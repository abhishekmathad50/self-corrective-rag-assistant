
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.ingestion import (
    ingest_documents,
    delete_document_from_vector_store,
)
from app.workflow import build_graph
from app.chat_store import (
    create_chat_session,
    get_chat_sessions,
    get_chat_messages,
    save_message,
    delete_chat_session,
    update_chat_title,
)


# =====================================================
# LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".md"}

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Self-corrective RAG assistant using LangGraph and FastAPI",
    version="1.6.2",
)

graph = build_graph()


# =====================================================
# REQUEST AND RESPONSE MODELS
# =====================================================

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    selected_documents: list[str] = Field(default_factory=list)

    # Retained for backward compatibility.
    # SQLite is the source of truth for chat history.
    chat_history: list[dict] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retry_count: int
    session_id: str


class CreateChatRequest(BaseModel):
    title: str = "New Chat"


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_available_documents():
    """Return supported documents available in the documents directory."""

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    return {
        file.name
        for file in DOCUMENTS_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS
    }


def validate_session(session_id: str):
    """Check whether a chat session exists."""

    sessions = get_chat_sessions()

    session_exists = any(
        chat["session_id"] == session_id
        for chat in sessions
    )

    if not session_exists:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )


# =====================================================
# ROOT / HEALTH CHECK
# =====================================================

@app.get("/")
def root():
    logger.info("Health check requested")

    return {
        "message": "RAG Technical Documentation Assistant is running",
        "version": app.version,
    }


# =====================================================
# CREATE CHAT
# =====================================================

@app.post("/chats")
def create_chat(request: CreateChatRequest):

    try:
        title = request.title.strip() or "New Chat"

        session_id = create_chat_session(title)

        logger.info("Chat session created: %s", session_id)

        return {
            "message": "Chat created successfully",
            "session_id": session_id,
            "title": title,
        }

    except Exception:
        logger.exception("Failed to create chat session")

        raise HTTPException(
            status_code=500,
            detail="Unable to create chat session.",
        )


# =====================================================
# LIST CHATS
# =====================================================

@app.get("/chats")
def list_chats():

    try:
        sessions = get_chat_sessions()

        logger.info(
            "Retrieved %s chat sessions",
            len(sessions),
        )

        return {
            "chats": sessions,
            "count": len(sessions),
        }

    except Exception:
        logger.exception("Failed to retrieve chat sessions")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve chat sessions.",
        )


# =====================================================
# RETRIEVE CHAT MESSAGES
# =====================================================

@app.get("/chats/{session_id}")
def retrieve_chat(session_id: str):

    try:
        validate_session(session_id)

        messages = get_chat_messages(session_id)

        logger.info(
            "Retrieved messages for chat: %s",
            session_id,
        )

        return {
            "session_id": session_id,
            "messages": messages,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to retrieve messages for session %s",
            session_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve chat messages.",
        )


# =====================================================
# DELETE CHAT
# =====================================================

@app.delete("/chats/{session_id}")
def remove_chat(session_id: str):

    try:
        deleted = delete_chat_session(session_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found.",
            )

        logger.info(
            "Chat session deleted: %s",
            session_id,
        )

        return {
            "message": "Chat deleted successfully",
            "session_id": session_id,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to delete chat: %s",
            session_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete chat session.",
        )


# =====================================================
# RAG QUERY
# =====================================================

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        # Validate selected documents
        available_documents = get_available_documents()

        invalid_documents = (
            set(request.selected_documents) - available_documents
        )

        if invalid_documents:
            raise HTTPException(
                status_code=400,
                detail="One or more selected documents are unavailable.",
            )

        # Create or validate chat session
        session_id = request.session_id

        if not session_id:
            session_id = create_chat_session()

        else:
            validate_session(session_id)

        # Retrieve previous messages from SQLite
        previous_messages = get_chat_messages(session_id)

        # Automatically generate chat title from first question
        if not previous_messages:

            title = question

            if len(title) > 40:
                title = title[:37].rstrip() + "..."

            update_chat_title(
                session_id=session_id,
                title=title,
            )

        # Prepare conversation history
        chat_history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in previous_messages
        ]

        # Save user question
        save_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        # Initialize LangGraph state
        initial_state = {
            "question": question,
            "rewritten_query": "",
            "selected_documents": request.selected_documents,
            "chat_history": chat_history,
            "documents": [],
            "relevant_documents": [],
            "answer": "",
            "retry_count": 0,
            "sources": [],
        }

        logger.info(
            "Processing RAG query for session %s",
            session_id,
        )

        # Execute RAG workflow
        result = graph.invoke(initial_state)

        answer = result["answer"]
        sources = result["sources"]
        retry_count = result["retry_count"]

        # Save assistant response
        save_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
            retry_count=retry_count,
        )

        logger.info(
            "Query completed. Session: %s, Sources: %s, Retries: %s",
            session_id,
            len(sources),
            retry_count,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            retry_count=retry_count,
            session_id=session_id,
        )

    except HTTPException:
        raise

    except ValueError as error:
        logger.warning("Invalid query request: %s", error)

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception:
        logger.exception(
            "RAG query failed for session %s",
            session_id if "session_id" in locals() else "unknown",
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please try again.",
        )


# =====================================================
# PDF UPLOAD
# =====================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    safe_filename = Path(file.filename).name

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    try:

        content = await file.read(MAX_UPLOAD_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB upload limit.",
            )

        if not content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            )

        DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = DOCUMENTS_DIR / safe_filename

        destination.write_bytes(content)

        logger.info(
            "PDF uploaded: %s",
            safe_filename,
        )

        return {
            "message": "PDF uploaded successfully. Run ingestion to index the document.",
            "filename": safe_filename,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to upload PDF: %s",
            safe_filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to upload the PDF.",
        )

    finally:
        await file.close()


# =====================================================
# DOCUMENT INGESTION
# =====================================================

@app.post("/ingest")
def ingest():

    try:
        logger.info("Document ingestion started")

        ingest_documents()

        logger.info(
            "Document ingestion completed successfully"
        )

        return {
            "message": "Documents ingested successfully",
        }

    except Exception:
        logger.exception("Document ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check backend logs for details.",
        )


# =====================================================
# LIST DOCUMENTS
# =====================================================

@app.get("/documents")
def get_documents():

    try:
        documents = sorted(get_available_documents())

        logger.info(
            "Retrieved %s available documents",
            len(documents),
        )

        return {
            "documents": documents,
            "count": len(documents),
        }

    except Exception:
        logger.exception("Failed to retrieve documents")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve documents.",
        )


# =====================================================
# DELETE DOCUMENT
# =====================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):

    try:

        # Prevent directory traversal
        if (
            Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename.",
            )

        if Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and Markdown documents can be deleted.",
            )

        document_path = DOCUMENTS_DIR / filename

        if not document_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        # Remove indexed chunks first
        deleted_chunks = delete_document_from_vector_store(filename)

        # Remove the original document
        document_path.unlink()

        logger.info(
            "Document deleted: %s. Chunks removed: %s",
            filename,
            deleted_chunks,
        )

        return {
            "message": "Document deleted successfully.",
            "filename": filename,
            "deleted_chunks": deleted_chunks,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to delete document: %s",
            filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete document. Check backend logs for details.",
        )


# =====================================================
# FEEDBACK
# =====================================================

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):

    if not request.feedback.strip():
        raise HTTPException(
            status_code=400,
            detail="Feedback cannot be empty.",
        )

    logger.info(
        "Feedback received successfully"
    )

    # Currently acknowledges feedback but does not persist it.

    return {
        "message": "Feedback received successfully",
        "feedback": request.feedback,
    }
