# RAG-Based Technical Documentation Assistant

A self-corrective Retrieval-Augmented Generation (RAG) system for answering questions from technical documentation.

The system uses **LangGraph** to orchestrate query analysis, document retrieval, LLM-based relevance grading, retry/rewrite logic, and grounded answer generation. A **FastAPI** application exposes the RAG pipeline through REST APIs.

---

## Features

- Query analysis and search-query rewriting
- Semantic document retrieval using ChromaDB
- Local sentence-transformer embeddings
- LLM-based document relevance grading
- Self-corrective retrieval with retry logic
- Grounded answer generation
- Source citations in generated answers
- FastAPI REST API
- Document ingestion endpoint
- Document listing endpoint
- Feedback endpoint
- Input validation and error handling
- Maximum retry limit to prevent infinite retrieval loops

---

## Architecture

The application follows a self-corrective RAG workflow implemented using LangGraph.

```text
User Question
      |
      v
+------------------+
|  Query Analysis  |
| Rewrite Question |
+--------+---------+
         |
         v
+------------------+
|    Retrieval     |
|    ChromaDB      |
+--------+---------+
         |
         v
+------------------+
| Document Grading |
|   LLM: YES/NO    |
+--------+---------+
         |
         v
   Relevant Docs?
      /       \
    YES        NO
     |          |
     v          v
 Generation   Rewrite
     |          |
     v          |
   Answer <-----+
     |
     v
   Sources