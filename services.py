"""Wrappers around the two external systems this app depends on:
Gemini (embeddings + text generation) and ChromaDB (vector storage).
Keeping them in small classes means the rest of the code never talks
to those libraries directly, and retries live in one place.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingService:
    """Turns text into vectors using Gemini's embedding model."""

    def __init__(self, client: genai.Client, model: str, max_retries: int = 4) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")

    def embed_documents_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_document(t) for t in texts]

    def _embed(self, text: str, task_type: str) -> list[float]:
        @retry(stop=stop_after_attempt(self._max_retries), wait=wait_exponential(min=1, max=10))
        def _call():
            result = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return result.embeddings[0].values

        return _call()


class LLMService:
    """Sends a prompt to Gemini and returns the generated text."""

    def __init__(self, client: genai.Client, model: str, max_retries: int = 4) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries

    def generate(self, prompt: str) -> str:
        @retry(stop=stop_after_attempt(self._max_retries), wait=wait_exponential(min=1, max=10))
        def _call():
            response = self._client.models.generate_content(model=self._model, contents=prompt)
            return response.text

        return _call()


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float


class VectorStore:
    """Stores and searches text chunks using ChromaDB."""

    def __init__(self, db_path: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def clear(self) -> None:
        """Delete all indexed materials (use this before starting a new subject/topic)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[str], embeddings: list[list[float]], source: str) -> int:
        ids = [hashlib.md5(f"{source}_{i}".encode()).hexdigest() for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]
        self._collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def sample(self, n: int) -> list[str]:
        docs = self._collection.get()["documents"]
        return docs[:n]

    def sample_random(self, n: int) -> list[str]:
        docs = self._collection.get()["documents"]
        return random.sample(docs, min(n, len(docs))) if docs else []

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        if not docs or not docs[0]:
            return []
        return [
            RetrievedChunk(text=d, source=m.get("source", "unknown"), distance=dist)
            for d, m, dist in zip(docs[0], metas[0], dists[0], strict=True)
        ]
    