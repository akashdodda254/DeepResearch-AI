from sentence_transformers import CrossEncoder
from typing import List, Dict


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(
            model_name,
            device="cpu"
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not documents:
            return []

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked_documents = []

        for document, score in zip(documents, scores):

            result = document.copy()
            result["rerank_score"] = float(score)

            reranked_documents.append(result)

        reranked_documents.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]