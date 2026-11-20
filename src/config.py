import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

try:
    import streamlit as st
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass


def get_llm():
    """Return a configured Gemini chat model with free-tier fallback."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to .env locally or Streamlit Secrets in cloud."
        )

    primary_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")

    primary = ChatGoogleGenerativeAI(
        model=primary_model,
        api_key=api_key,
        timeout=30,
        max_retries=2,
    )

    if fallback_model and fallback_model != primary_model:
        fallback = ChatGoogleGenerativeAI(
            model=fallback_model,
            api_key=api_key,
            timeout=30,
            max_retries=2,
        )
        return primary.with_fallbacks([fallback])

    return primary