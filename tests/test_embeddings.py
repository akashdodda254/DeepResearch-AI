from retrieval.embedding_model import EmbeddingModel


def test_embedding_generation():
    print("Loading embedding model...")

    model = EmbeddingModel()

    texts = [
        "Retrieval Augmented Generation combines retrieval with language models.",
        "FAISS enables efficient similarity search over vector embeddings.",
        "Machine learning models learn patterns from training data."
    ]

    embeddings = model.generate_embeddings(texts)

    print("\nEMBEDDING TEST SUCCESSFUL")
    print("-------------------------")
    print("Number of texts:", len(texts))
    print("Embedding shape:", embeddings.shape)
    print("Vector dimension:", embeddings.shape[1])

    query = "How does RAG retrieve relevant information?"

    query_embedding = model.generate_query_embedding(query)

    print("Query embedding shape:", query_embedding.shape)

    assert embeddings.shape == (3, 384)
    assert query_embedding.shape == (1, 384)

    print("\nAll checks passed!")


if __name__ == "__main__":
    test_embedding_generation()