"""Embedding generation using Google Gemini."""

from typing import List
from google import genai


def get_embedding(text: str, client: genai.Client, model: str = "gemini-embedding-001") -> List[float]:
    """Generate embedding for a single text.

    Args:
        text: Text to embed.
        client: Gemini client instance.
        model: Embedding model name.

    Returns:
        Embedding vector as list of floats.
    """
    resp = client.models.embed_content(model=model, contents=text)
    return resp.embeddings[0].values


def embed_documents(
    docs: dict[str, str], client: genai.Client, model: str = "gemini-embedding-001"
) -> dict[str, List[float]]:
    """Generate embeddings for all documents.

    Args:
        docs: Dictionary mapping doc names to content.
        client: Gemini client instance.
        model: Embedding model name.

    Returns:
        Dictionary mapping doc names to embedding vectors.
    """
    return {name: get_embedding(text, client, model) for name, text in docs.items()}