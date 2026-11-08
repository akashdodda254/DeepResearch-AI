from rank_bm25 import BM25Okapi
from typing import List, Dict
import re


class BM25Retriever:
    """Keyword-based document retrieval using BM25."""

    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

    def tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase tokens."""

        return re.findall(r"\b\w+\b", text.lower())

    def index_documents(self, documents: List[Dict]):
        """Build BM25 index from document chunks."""

        if not documents:
            raise ValueError("Documents cannot be empty.")

        self.documents = documents

        self.tokenized_corpus = [
            self.tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using BM25."""

        if self.bm25 is None:
            raise ValueError("Index documents before searching.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        tokenized_query = self.tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            result = self.documents[index].copy()
            result["bm25_score"] = float(scores[index])

            results.append(result)

        return results