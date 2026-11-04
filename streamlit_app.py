
import requests
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="Self-Corrective RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# -----------------------------
# API Helpers
# -----------------------------
def fetch_documents():
    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.RequestException:
        return None


def fetch_chats():
    try:
        response = requests.get(f"{API_URL}/chats", timeout=10)
        response.raise_for_status()
        return response.json().get("chats", [])
    except requests.RequestException:
        return None


def fetch_chat_messages(session_id):
    response = requests.get(
        f"{API_URL}/chats/{session_id}", timeout=10
    )
    response.raise_for_status()
    return response.json().get("messages", [])


def create_chat():
    response = requests.post(
        f"{API_URL}/chats",
        json={"title": "New Chat"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["session_id"]


def delete_chat(session_id):
    response = requests.delete(
        f"{API_URL}/chats/{session_id}", timeout=10
    )
    response.raise_for_status()


def delete_document(filename):
    response = requests.delete(
        f"{API_URL}/documents/{filename}", timeout=30
    )
    response.raise_for_status()
    return response.json()


def display_sources(sources):
    if sources:
        st.markdown("### 📚 Source References")
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**[{index}]** `{source}`")


# -----------------------------
# Session State Initialization
# -----------------------------
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_document_delete" not in st.session_state:
    st.session_state.pending_document_delete = None


# -----------------------------
# Chat Session Management
# -----------------------------
st.sidebar.title("💬 Chat History")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    try:
        st.session_state.active_chat_id = create_chat()
        st.session_state.messages = []
        st.rerun()
    except requests.RequestException as error:
        st.sidebar.error(f"Could not create chat: {error}")


saved_chats = fetch_chats()

if saved_chats is None:
    st.sidebar.error("Could not load saved conversations.")
    saved_chats = []

st.sidebar.markdown("**Previous Conversations**")

for chat in saved_chats:
    session_id = chat["session_id"]
    title = chat["title"]

    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        if st.button(
            title,
            key=f"open_{session_id}",
            use_container_width=True,
        ):
            try:
                st.session_state.active_chat_id = session_id
                st.session_state.messages = fetch_chat_messages(session_id)
                st.rerun()
            except requests.RequestException as error:
                st.sidebar.error(f"Could not open chat: {error}")

    with col2:
        if st.button("🗑️", key=f"delete_{session_id}"):
            try:
                delete_chat(session_id)

                if st.session_state.active_chat_id == session_id:
                    st.session_state.active_chat_id = None
                    st.session_state.messages = []

                st.rerun()
            except requests.RequestException as error:
                st.sidebar.error(f"Could not delete chat: {error}")


st.sidebar.markdown("---")


# -----------------------------
# Backend Status
# -----------------------------
st.sidebar.header("⚙️ Backend Status")

if st.sidebar.button("Check Backend"):
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        response.raise_for_status()
        st.sidebar.success("Backend is running!")
    except requests.RequestException:
        st.sidebar.error("Backend is not reachable.")


# -----------------------------
# PDF Upload
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload PDF")

uploaded_pdf = st.sidebar.file_uploader(
    "Choose a technical PDF",
    type=["pdf"],
)

if uploaded_pdf is not None:
    if st.sidebar.button("Upload and Ingest PDF"):
        try:
            upload_response = requests.post(
                f"{API_URL}/upload",
                files={
                    "file": (
                        uploaded_pdf.name,
                        uploaded_pdf.getvalue(),
                        "application/pdf",
                    )
                },
                timeout=60,
            )
            upload_response.raise_for_status()

            with st.spinner("Ingesting documents..."):
                ingest_response = requests.post(
                    f"{API_URL}/ingest",
                    timeout=180,
                )
                ingest_response.raise_for_status()

            st.sidebar.success("PDF uploaded and ingested!")
            st.rerun()

        except requests.RequestException as error:
            st.sidebar.error(f"Upload or ingestion failed: {error}")


# -----------------------------
# Document Management
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Document Management")

available_documents = fetch_documents()

if available_documents is None:
    st.sidebar.error("Could not load documents.")
    available_documents = []

if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = available_documents

# Remove selections for documents no longer available
st.session_state.selected_documents = [
    name
    for name in st.session_state.selected_documents
    if name in available_documents
]

if st.sidebar.button("🔄 Refresh Documents"):
    st.session_state.selected_documents = available_documents
    st.session_state.pending_document_delete = None
    st.rerun()

if available_documents:
    st.sidebar.caption(f"{len(available_documents)} document(s) available")

    for index, filename in enumerate(available_documents):
        col1, col2 = st.sidebar.columns([4, 1])

        with col1:
            st.sidebar.write(f"📄 {filename}")

        with col2:
            if st.button(
                "🗑️",
                key=f"delete_document_{index}_{filename}",
                help=f"Delete {filename}",
            ):
                st.session_state.pending_document_delete = filename
                st.rerun()

else:
    st.sidebar.info("No documents found. Upload a PDF to get started.")

# Confirmation before deletion
pending_delete = st.session_state.pending_document_delete

if pending_delete:
    st.sidebar.warning(
        f"Are you sure you want to delete **{pending_delete}**? "
        "This also removes its indexed chunks."
    )

    confirm_col, cancel_col = st.sidebar.columns(2)

    with confirm_col:
        if st.button("Yes, Delete", key="confirm_document_delete"):
            try:
                with st.spinner("Deleting document..."):
                    result = delete_document(pending_delete)

                st.session_state.selected_documents = [
                    name
                    for name in st.session_state.selected_documents
                    if name != pending_delete
                ]

                st.session_state.pending_document_delete = None

                st.sidebar.success(
                    f"Deleted {pending_delete}. "
                    f"Removed {result.get('deleted_chunks', 0)} chunks."
                )
                st.rerun()

            except requests.RequestException as error:
                st.sidebar.error(f"Document deletion failed: {error}")

    with cancel_col:
        if st.button("Cancel", key="cancel_document_delete"):
            st.session_state.pending_document_delete = None
            st.rerun()


# -----------------------------
# Document Selection
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Document Selection")

selected_documents = st.sidebar.multiselect(
    "Select documents to search",
    options=available_documents,
    key="selected_documents",
    help="Leave empty to search all available documents.",
)


# -----------------------------
# Main Chat Interface
# -----------------------------
st.title("🤖 Self-Corrective RAG Assistant")

st.markdown(
    "Upload technical documents and ask questions "
    "using your self-corrective RAG workflow."
)

if st.session_state.active_chat_id is None:
    st.info(
        "Select a previous conversation or click New Chat to begin."
    )

else:
    st.caption(f"Active chat: {st.session_state.active_chat_id}")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                display_sources(message["sources"])

            if "retry_count" in message:
                st.caption(f"Retry count: {message['retry_count']}")

    # Ask a new question
    question = st.chat_input(
        "Ask a question about your technical documents..."
    )

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching documents and generating an answer..."
                ):
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": question,
                            "session_id": st.session_state.active_chat_id,
                            "selected_documents": selected_documents,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    result = response.json()

                answer = result["answer"]
                sources = result.get("sources", [])
                retry_count = result.get("retry_count", 0)

                st.markdown(answer)
                display_sources(sources)
                st.caption(f"Retry count: {retry_count}")

                st.session_state.active_chat_id = result["session_id"]

                st.session_state.messages.extend([
                    {
                        "role": "user",
                        "content": question,
                        "sources": [],
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retry_count": retry_count,
                    },
                ])

                st.rerun()

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running on port 8000."
                )

            except requests.Timeout:
                st.error("The request timed out.")

            except requests.HTTPError as error:
                st.error(f"Backend error: {error}")
                if error.response is not None:
                    st.code(error.response.text)

            except requests.RequestException as error:
                st.error(f"Request failed: {error}")
