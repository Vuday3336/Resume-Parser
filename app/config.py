"""Centralized environment configuration. All secrets come from .env / hosting env vars — never hardcode keys."""
import os

from dotenv import load_dotenv

load_dotenv()


def _load_streamlit_secrets() -> None:
    """Streamlit Community Cloud injects secrets via st.secrets, not the process environment.
    Mirror them into os.environ on import so the rest of the app can stay framework-agnostic."""
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            os.environ.setdefault(key, str(value))
    except Exception:
        pass  # not running under Streamlit, or no secrets.toml configured — fine for local/CLI use


_load_streamlit_secrets()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    SHORTLIST_SIZE: int = int(os.getenv("SHORTLIST_SIZE", "10"))  # candidates that reach the (expensive) LLM rerank stage

    def require_openai(self) -> None:
        if not self.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or hosting secrets."
            )

    def require_database(self) -> None:
        if not self.DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Add your Supabase Postgres connection string to .env."
            )


settings = Settings()
