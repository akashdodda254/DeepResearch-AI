from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    """Generate semantic embeddings for research-paper text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Convert a list of text passages into numerical vectors."""

        if not texts:
            raise ValueError("Text list cannot be empty.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate an embedding for a search query."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding