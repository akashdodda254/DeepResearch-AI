from retrieval.bm25_retriever import BM25Retriever


def main():
    documents = [
        {
            "chunk_id": 0,
            "paper_id": "paper_001",
            "page_number": 1,
            "text": "Retrieval Augmented Generation combines information retrieval with large language models."
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

    retriever = BM25Retriever()

    # Build keyword search index
    retriever.index_documents(documents)

    print("\nBM25 INDEX CREATED")
    print("------------------")
    print("Total documents:", len(documents))

    # Search
    query = "Retrieval Augmented Generation"

    results = retriever.search(
        query=query,
        top_k=2
    )

    print("\nBM25 SEARCH RESULTS")
    print("-------------------")

    for result in results:
        print("\nPaper ID:", result["paper_id"])
        print("Page:", result["page_number"])
        print("BM25 Score:", round(result["bm25_score"], 4))
        print("Text:", result["text"])

    assert results[0]["paper_id"] == "paper_001"

    print("\nBM25 TEST PASSED!")


if __name__ == "__main__":
    main()