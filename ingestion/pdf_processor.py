import pymupdf
import requests

from pathlib import Path
from typing import Optional


class PDFProcessor:
    """Downloads academic PDFs and extracts structured text."""

    def __init__(self, storage_dir: str = "data/papers"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download an academic paper PDF."""

        safe_paper_id = paper_id.replace("/", "_")
        pdf_path = self.storage_dir / f"{safe_paper_id}.pdf"

        if pdf_path.exists():
            print(f"PDF already exists: {pdf_path}")
            return pdf_path

        try:
            response = requests.get(
                pdf_url,
                timeout=60,
                headers={"User-Agent": "DeepResearch-AI/1.0"}
            )
            response.raise_for_status()

            if not response.content.startswith(b"%PDF-"):
                raise ValueError("Downloaded file is not a valid PDF.")

            pdf_path.write_bytes(response.content)

            print(f"Downloaded: {pdf_path}")
            return pdf_path

        except (requests.RequestException, ValueError) as error:
            print(f"PDF download failed: {error}")
            return None

    def extract_text(self, pdf_path: Path) -> dict:
        """Extract text while preserving page numbers."""

        pages = []

        try:
            with pymupdf.open(pdf_path) as document:

                metadata = document.metadata

                for page_number, page in enumerate(document, start=1):

                    text = page.get_text("text").strip()

                    if text:
                        pages.append({
                            "page_number": page_number,
                            "text": text
                        })

                return {
                    "file_name": pdf_path.name,
                    "total_pages": len(document),
                    "title": metadata.get("title"),
                    "author": metadata.get("author"),
                    "pages": pages
                }

        except (pymupdf.FileDataError, OSError) as error:
            print(f"PDF extraction failed: {error}")
            return {}

    def process_paper(self, pdf_url: str, paper_id: str) -> dict:
        """Download and extract an academic paper."""

        pdf_path = self.download_pdf(pdf_url, paper_id)

        if pdf_path is None:
            return {}

        return self.extract_text(pdf_path)