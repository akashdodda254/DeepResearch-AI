from typing import List, Dict


class DocumentChunker:
    """Split research papers into smaller overlapping text chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_chunks(
        self,
        pages: List[Dict],
        paper_id: str
    ) -> List[Dict]:

        chunks = []
        chunk_id = 0

        step = self.chunk_size - self.overlap

        for page in pages:
            words = page["text"].split()
            page_number = page["page_number"]

            for start in range(0, len(words), step):
                chunk_words = words[start:start + self.chunk_size]

                if not chunk_words:
                    continue

                chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "text": " ".join(chunk_words),
                    "word_count": len(chunk_words)
                })

                chunk_id += 1

        return chunks