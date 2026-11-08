import faiss
import numpy as np
import json

from pathlib import Path
from typing import List, Dict


class FAISSVectorStore:
    """Store and search document embeddings using FAISS."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict]
    ):
        """Add document embeddings and metadata to FAISS."""

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimensions do not match.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid values.")

        self.index.add(embeddings)
        self.metadata.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Find the most relevant document chunks."""

        if self.index.ntotal == 0:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension does not match.")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index_id in zip(scores[0], indices[0]):

            if index_id == -1:
                continue

            result = self.metadata[index_id].copy()
            result["similarity_score"] = float(score)

            results.append(result)

        return results

    def save(self, directory: str = "data/vector_store"):
        """Save FAISS index and document metadata."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(self.metadata, file, indent=4)

        print(f"Vector store saved to: {directory}")

    def load(self, directory: str = "data/vector_store"):
        """Load FAISS index and document metadata."""

        directory = Path(directory)

        self.index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            "r",
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        self.dimension = self.index.d

        print(f"Vector store loaded from: {directory}")