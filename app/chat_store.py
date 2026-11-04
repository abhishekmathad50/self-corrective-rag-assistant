
import sqlite3
import json
import uuid

from pathlib import Path
from datetime import datetime, timezone


# Database configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "data"
DATABASE_PATH = DATABASE_DIR / "chat_history.db"


def get_connection():
    """Create a SQLite database connection."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create chat session and message tables."""

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                retry_count INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id)
                    REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE
            )
        """)

        connection.commit()


def create_chat_session(title="New Chat"):
    """Create a new chat session."""

    session_id = str(uuid.uuid4())
    current_time = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO chat_sessions
            (session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                title,
                current_time,
                current_time,
            ),
        )

    return session_id


def update_chat_title(session_id, title):
    """Update the title of an existing chat session."""

    current_time = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE chat_sessions
            SET title = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (title, current_time, session_id),
        )

        connection.commit()


def get_chat_sessions():
    """Return all chat sessions, newest first."""

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def save_message(
    session_id,
    role,
    content,
    sources=None,
    retry_count=0,
):
    """Save a message to an existing chat session."""

    current_time = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        session = connection.execute(
            """
            SELECT session_id
            FROM chat_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        if session is None:
            raise ValueError("Chat session not found.")

        connection.execute(
            """
            INSERT INTO chat_messages
            (session_id, role, content, sources,
             retry_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                json.dumps(sources or []),
                retry_count,
                current_time,
            ),
        )

        connection.execute(
            """
            UPDATE chat_sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (current_time, session_id),
        )


def get_chat_messages(session_id):
    """Retrieve all messages from a chat session."""

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT role, content, sources, retry_count, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    messages = []

    for row in rows:

        message = dict(row)
        message["sources"] = json.loads(
            message["sources"] or "[]"
        )

        messages.append(message)

    return messages


def delete_chat_session(session_id):
    """Delete a chat session and its messages."""

    with get_connection() as connection:

        connection.execute("PRAGMA foreign_keys = ON")

        cursor = connection.execute(
            """
            DELETE FROM chat_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )

        connection.commit()

        return cursor.rowcount > 0


# Initialize database when this module is imported
initialize_database()
