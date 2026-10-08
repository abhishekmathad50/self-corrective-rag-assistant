
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingestion import ingest_documents
from app.workflow import build_graph


DOCUMENTS_DIR = Path("data/documents")


app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Self-corrective RAG assistant using LangGraph and FastAPI",
    version="1.0.0",
)


# Build the LangGraph workflow
graph = build_graph()


# -----------------------------
# Request / Response Models
# -----------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retry_count: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str


# -----------------------------
# Root Endpoint
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "RAG Technical Documentation Assistant is running"
    }


# -----------------------------
# Query Endpoint
# -----------------------------

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    initial_state = {
        "question": request.question,
        "rewritten_query": "",
        "documents": [],
        "relevant_documents": [],
        "answer": "",
        "retry_count": 0,
        "sources": [],
    }

    try:
        result = graph.invoke(initial_state)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retry_count=result["retry_count"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


# -----------------------------
# Ingest Endpoint
# -----------------------------

@app.post("/ingest")
def ingest():

    try:
        ingest_documents()

        return {
            "message": "Documents ingested successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during ingestion: {str(e)}",
        )


# -----------------------------
# Documents Endpoint
# -----------------------------

@app.get("/documents")
def get_documents():

    try:
        documents = [
            file.name
            for file in DOCUMENTS_DIR.glob("*.md")
        ]

        return {
            "documents": documents,
            "count": len(documents),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}",
        )


# -----------------------------
# Feedback Endpoint
# -----------------------------

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):

    if not request.feedback.strip():
        raise HTTPException(
            status_code=400,
            detail="Feedback cannot be empty",
        )

    return {
        "message": "Feedback received successfully",
        "feedback": request.feedback,
    }