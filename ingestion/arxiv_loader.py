import arxiv
from typing import List
from pydantic import BaseModel, HttpUrl


class ResearchPaper(BaseModel):
    """Structured representation of an academic research paper."""

    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    pdf_url: HttpUrl
    categories: List[str]


class ArxivPaperLoader:
    """Search and retrieve relevant academic papers from arXiv."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

        self.client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,
            num_retries=3
        )

    def build_search_query(self, query: str) -> str:
        """Convert research questions into focused arXiv queries."""

        question = query.lower()

        # Medication information retrieval
        if any(term in question for term in [
            "medication",
            "formulary",
            "drug information"
        ]):
            return (
                'all:"retrieval augmented generation" '
                'AND (all:medication OR all:formulary OR all:drug)'
            )

        # Medical question answering
        if any(term in question for term in [
            "question answering",
            "medical questions"
        ]):
            return (
                'all:"retrieval augmented generation" '
                'AND (all:"question answering" OR all:medical)'
            )

        # Clinical and medical applications
        if any(term in question for term in [
            "clinical",
            "clinical applications",
            "medical literature"
        ]):
            return (
                'all:"retrieval augmented generation" '
                'AND (all:clinical OR all:PubMed OR all:medical)'
            )

        if "healthcare" in question or "medical" in question:
            return (
                'all:"retrieval augmented generation" '
                'AND (all:healthcare OR all:medical OR all:clinical)'
            )

        return f'all:"{query.strip()}"'

    def search_papers(self, query: str) -> List[ResearchPaper]:
        """Search arXiv using a focused research query."""

        search_query = self.build_search_query(query)

        print(f"arXiv search query: {search_query}")

        search = arxiv.Search(
            query=search_query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []

        try:
            results = self.client.results(search)

            for result in results:

                paper = ResearchPaper(
                    paper_id=result.get_short_id(),
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    published=result.published.strftime("%Y-%m-%d"),
                    pdf_url=str(result.pdf_url),
                    categories=result.categories
                )

                papers.append(paper)

            return papers

        except Exception as error:
            print(f"Error retrieving papers: {error}")
            return []


if __name__ == "__main__":

    loader = ArxivPaperLoader(max_results=5)

    query = (
        "How is Retrieval-Augmented Generation "
        "used in healthcare?"
    )

    papers = loader.search_papers(query)

    print(f"\nFound {len(papers)} research papers\n")

    for index, paper in enumerate(papers, start=1):

        print(f"\nPaper {index}")
        print("-" * 60)

        print("Title:", paper.title)
        print("Authors:", ", ".join(paper.authors))
        print("Published:", paper.published)
        print("Paper ID:", paper.paper_id)
        print("Abstract:", paper.abstract[:300])
        print("PDF:", paper.pdf_url)