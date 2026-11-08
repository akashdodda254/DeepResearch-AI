from typing import List, Dict

from retrieval.vector_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.embedding_model import EmbeddingModel


class HybridRetriever:
    """Combine FAISS and BM25 using Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:

        # FAISS semantic search
        query_embedding = (
            self.embedding_model.generate_query_embedding(query)
        )

        faiss_results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # BM25 keyword search
        bm25_results = self.bm25_retriever.search(
            query,
            top_k=top_k
        )

        # Reciprocal Rank Fusion
        fused_results = {}

        for results in [faiss_results, bm25_results]:

            for rank, document in enumerate(results, start=1):

                key = (
                    document["paper_id"],
                    document["chunk_id"]
                )

                if key not in fused_results:
                    fused_results[key] = document.copy()
                    fused_results[key]["rrf_score"] = 0.0

                fused_results[key]["rrf_score"] += (
                    1 / (self.rrf_k + rank)
                )

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        return ranked_results[:top_k]