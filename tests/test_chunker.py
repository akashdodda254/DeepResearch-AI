from ingestion.arxiv_loader import ArxivPaperLoader
from ingestion.pdf_processor import PDFProcessor
from ingestion.chunker import DocumentChunker


def main():
    loader = ArxivPaperLoader(max_results=1)
    papers = loader.search_papers("Retrieval Augmented Generation")

    if not papers:
        print("No papers found.")
        return

    paper = papers[0]

    processor = PDFProcessor()
    result = processor.process_paper(
        pdf_url=str(paper.pdf_url),
        paper_id=paper.paper_id
    )

    if not result:
        print("PDF processing failed.")
        return

    chunker = DocumentChunker(chunk_size=500, overlap=100)

    chunks = chunker.split_into_chunks(
        pages=result["pages"],
        paper_id=paper.paper_id
    )

    print("\nCHUNKING SUCCESSFUL")
    print("-------------------")
    print("Paper:", paper.title)
    print("Total chunks:", len(chunks))

    for chunk in chunks[:3]:
        print("\nChunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page_number"])
        print("Word count:", chunk["word_count"])
        print("Text preview:", chunk["text"][:200])


if __name__ == "__main__":
    main()