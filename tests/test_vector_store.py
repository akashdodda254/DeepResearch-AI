from retrieval.embedding_model import EmbeddingModel
from retrieval.vector_store import FAISSVectorStore


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

    # Initialize FAISS vector store
    vector_store = FAISSVectorStore(dimension=384)

    # Add documents
    vector_store.add_documents(embeddings, documents)

    print("\nDOCUMENTS ADDED TO FAISS")
    print("------------------------")
    print("Total vectors:", vector_store.index.ntotal)

    # Search for relevant documents
    query = "How does retrieval augmented generation work?"

    query_embedding = embedding_model.generate_query_embedding(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=2
    )

    print("\nSEARCH RESULTS")
    print("--------------")

    for result in results:
        print("\nPaper ID:", result["paper_id"])
        print("Page:", result["page_number"])
        print("Similarity:", round(result["similarity_score"], 4))
        print("Text:", result["text"])

    # Save vector database
    vector_store.save()

    print("\nFAISS TEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()