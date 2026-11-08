from retrieval.embedding_model import EmbeddingModel
from retrieval.vector_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever


def main():

    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    # Sample research-paper chunks
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

    # Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever
    )

    # Perform hybrid search
    query = "How does Retrieval Augmented Generation retrieve information?"

    results = hybrid_retriever.search(
        query=query,
        top_k=3
    )

    print("\nHYBRID RETRIEVAL RESULTS")
    print("-----------------------")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Paper ID:", result["paper_id"])
        print("Page:", result["page_number"])
        print("RRF Score:", round(result["rrf_score"], 5))
        print("Text:", result["text"])

    assert results
    assert results[0]["paper_id"] == "paper_001"

    print("\nHYBRID RETRIEVAL TEST PASSED!")


if __name__ == "__main__":
    main()