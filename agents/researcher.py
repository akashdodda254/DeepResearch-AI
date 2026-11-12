from ingestion.arxiv_loader import ArxivPaperLoader
from ingestion.pdf_processor import PDFProcessor
from ingestion.chunker import DocumentChunker

from retrieval.embedding_model import EmbeddingModel
from retrieval.vector_store import FAISSVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


class ResearcherAgent:
    """Search academic papers and retrieve relevant evidence."""

    def __init__(self):
        self.arxiv = ArxivPaperLoader(max_results=3)
        self.pdf_processor = PDFProcessor()
        self.chunker = DocumentChunker()

        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSVectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker()

    def research(
        self,
        question: str,
        max_results: int = 3,
        retrieval_top_k: int = 3,
        rerank_top_k: int = 1,
        return_stages: bool = False
    ):
        """
        Retrieve academic evidence for a research question.

        Parameters:
            question: Research question.
            max_results: Maximum number of arXiv papers to process.
            retrieval_top_k: Number of chunks retrieved by hybrid search.
            rerank_top_k: Number of chunks returned after reranking.
        """

        if max_results < 1 or retrieval_top_k < 1 or rerank_top_k < 1:
            raise ValueError("All retrieval limits must be at least 1.")

        print(f"\nResearching: {question}")
        
        # Clear previous question's retrieval indexes
        self.vector_store.reset()
        self.bm25.reset()

        # 1. Search arXiv
        self.arxiv.max_results = max_results
        papers = self.arxiv.search_papers(question)

        if not papers:
            print("No papers found.")
            return []
        
        candidate_paper_ids = [
            paper.paper_id.split("v")[0]
            for paper in papers
        ]

        all_chunks = []

        # 2. Download PDFs, extract text and create chunks
        for paper in papers:
            print(f"\nProcessing: {paper.title}")

            extracted = self.pdf_processor.process_paper(
                str(paper.pdf_url),
                paper.paper_id
            )

            if not extracted or not extracted.get("pages"):
                print("Could not extract paper text.")
                continue

            chunks = self.chunker.split_into_chunks(
                extracted["pages"],
                paper.paper_id
            )

            for chunk in chunks:
                chunk["title"] = paper.title
                chunk["authors"] = paper.authors
                chunk["published"] = paper.published
                chunk["pdf_url"] = str(paper.pdf_url)

            all_chunks.extend(chunks)

        if not all_chunks:
            print("No text chunks available.")
            return []

        print(f"\nCreated {len(all_chunks)} text chunks.")

        # 3. Generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]

        embeddings = self.embedding_model.generate_embeddings(texts)

        # 4. Build FAISS and BM25 indexes
        self.vector_store.add_documents(
            embeddings,
            all_chunks
        )

        self.bm25.index_documents(all_chunks)

        # 5. Hybrid retrieval
        hybrid = HybridRetriever(
            self.embedding_model,
            self.vector_store,
            self.bm25
        )

        hybrid_results = hybrid.search(
            question,
            top_k=retrieval_top_k
        )

        # 6. Cross-encoder reranking
        reranked_results = self.reranker.rerank(
            question,
            hybrid_results,
            top_k=rerank_top_k
        )

# Return detailed results when stage evaluation is requested.
        if return_stages:
            return {
                "candidate_paper_ids": candidate_paper_ids,
                "hybrid_results": hybrid_results,
                "reranked_results": reranked_results,
            }

        return reranked_results


if __name__ == "__main__":
    researcher = ResearcherAgent()

    question = "How is Retrieval-Augmented Generation used in healthcare?"

    results = researcher.research(question)

    print("\nTOP RESEARCH EVIDENCE")
    print("=" * 60)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Paper:", result["title"])
        print("Page:", result["page_number"])
        print("Score:", result["rerank_score"])
        print("Text:", result["text"][:500])