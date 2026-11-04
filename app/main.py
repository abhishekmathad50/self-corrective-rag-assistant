
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

DOCUMENTS_DIR = Path("data/documents")

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

        logger.info("Retrieved %s chat sessions", len(sessions))

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

        messages = get_chat_messages(session_id)

        logger.info("Retrieved messages for chat: %s", session_id)

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

        logger.info("Chat session deleted: %s", session_id)

        return {
            "message": "Chat deleted successfully",
            "session_id": session_id,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to delete chat: %s", session_id)

        raise HTTPException(
            status_code=500,
            detail="Unable to delete chat session.",
        )


# =====================================================
# RAG QUERY
# =====================================================

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    available_documents = {
        file.name
        for file in DOCUMENTS_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in {".md", ".pdf"}
    }

    invalid_documents = (
        set(request.selected_documents) - available_documents
    )

    if invalid_documents:
        raise HTTPException(
            status_code=400,
            detail="One or more selected documents are unavailable.",
        )

    session_id = request.session_id

    try:
        if not session_id:
            session_id = create_chat_session()

        previous_messages = get_chat_messages(session_id)

        # Automatically create chat title from first question
        if not previous_messages:
            title = request.question.strip()

            if len(title) > 40:
                title = title[:37].rstrip() + "..."

            update_chat_title(
                session_id=session_id,
                title=title,
            )

        chat_history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in previous_messages
        ]

        save_message(
            session_id=session_id,
            role="user",
            content=request.question,
        )

        initial_state = {
            "question": request.question,
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

        result = graph.invoke(initial_state)

        answer = result["answer"]
        sources = result["sources"]
        retry_count = result["retry_count"]

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

    except Exception:
        logger.exception(
            "RAG query failed for session %s",
            session_id,
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
        content = await file.read()

        if not content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            )

        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        destination = DOCUMENTS_DIR / safe_filename
        destination.write_bytes(content)

        logger.info("PDF uploaded: %s", safe_filename)

        return {
            "message": "PDF uploaded successfully",
            "filename": safe_filename,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Failed to upload PDF: %s", safe_filename)

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

        logger.info("Document ingestion completed successfully")

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

    print("DOCUMENT ENDPOINT WAS CALLED", flush=True)

    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        documents = sorted(
            file.name
            for file in DOCUMENTS_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() in {".md", ".pdf"}
        )

        logger.info(
            "Retrieved %s available documents",
            len(documents),
        )

        print(
            f"DOCUMENT COUNT: {len(documents)}",
            flush=True,
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
        if Path(filename).name != filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename.",
            )

        if filename in {".", ".."}:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename.",
            )

        if Path(filename).suffix.lower() not in {".pdf", ".md"}:
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

        deleted_chunks = delete_document_from_vector_store(filename)

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
        logger.exception("Failed to delete document: %s", filename)

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

    logger.info("Feedback received successfully")

    return {
        "message": "Feedback received successfully",
        "feedback": request.feedback,
    }
