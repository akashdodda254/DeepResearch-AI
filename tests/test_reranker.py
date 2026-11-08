from retrieval.embedding_model import EmbeddingModel
from retrieval.vector_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


def main():

    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    documents = [
        {
            "chunk_id": 0,
            "paper_id": "paper_001",
            "page_number": 1,
            "text": "Retrieval Augmented Generation combines information retrieval with large language models to generate grounded answers."
        },
        {
            "chunk_id": 1,
            "paper_id": "paper_002",
            "page_number": 3,
            "text": "FAISS is a library for efficient similarity search and clustering of dense vectors."
        },
        {
            "chunk_id": 2,
            "paper_id": "paper_003",
            "page_number": 5,
            "text": "Neural networks learn complex patterns from training data using optimization algorithms."
        }
    ]

    # Generate embeddings
    texts = [document["text"] for document in documents]
    embeddings = embedding_model.generate_embeddings(texts)

    # Initialize FAISS
    vector_store = FAISSVectorStore(dimension=384)
    vector_store.add_documents(embeddings, documents)

    # Initialize BM25
    bm25_retriever = BM25Retriever()
    bm25_retriever.index_documents(documents)

    # Initialize hybrid retrieval
    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever
    )

    query = "How does Retrieval Augmented Generation retrieve information?"

    hybrid_results = hybrid_retriever.search(
        query=query,
        top_k=3
    )

    print("\nHYBRID RETRIEVAL COMPLETED")

    # Initialize reranker
    print("\nLoading cross-encoder model...")

    reranker = CrossEncoderReranker()

    # Rerank results
    reranked_results = reranker.rerank(
        query=query,
        documents=hybrid_results,
        top_k=3
    )

    print("\nRERANKED RESULTS")
    print("----------------")

    for rank, result in enumerate(reranked_results, start=1):
        print(f"\nRank: {rank}")
        print("Paper ID:", result["paper_id"])
        print("Page:", result["page_number"])
        print("Rerank Score:", round(result["rerank_score"], 4))
        print("Text:", result["text"])

    assert len(reranked_results) == 3
    assert all("rerank_score" in result for result in reranked_results)

    print("\nCROSS-ENCODER RERANKING TEST PASSED!")


if __name__ == "__main__":
    main()