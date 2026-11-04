import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"
TEMPERATURE = 0


def get_llm():
    """
    Initialize and return the Groq language model.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. Please configure it in your .env file."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=GROQ_API_KEY,
    )
