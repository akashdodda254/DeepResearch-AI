from ingestion.arxiv_loader import ArxivPaperLoader
from ingestion.pdf_processor import PDFProcessor


def main():

    loader = ArxivPaperLoader(max_results=1)

    papers = loader.search_papers("Retrieval Augmented Generation")

    if not papers:
        print("No papers retrieved.")
        return

    paper = papers[0]

    processor = PDFProcessor()

    result = processor.process_paper(
        pdf_url=str(paper.pdf_url),
        paper_id=paper.paper_id
    )

    if not result:
        print("Paper processing failed.")
        return

    print("\nPDF PROCESSING SUCCESSFUL")
    print("-------------------------")

    print("Title:", result["title"])
    print("Author:", result["author"])
    print("Total pages:", result["total_pages"])

    for page in result["pages"][:2]:
        print(f"\nPage {page['page_number']}:")
        print(page["text"][:500])


if __name__ == "__main__":
    main()