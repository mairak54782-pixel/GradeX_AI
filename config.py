"""All app settings in one place. Values come from environment variables / .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Copy .env.example to .env and add your key."
            )

        self.generation_model: str = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

        self.vector_db_path: Path = Path(os.getenv("VECTOR_DB_PATH", "./gradex_db"))
        self.collection_name: str = os.getenv("COLLECTION_NAME", "teacher_patterns")
        self.top_k: int = int(os.getenv("TOP_K", "5"))
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "4"))


def get_settings() -> Settings:
    return Settings()
