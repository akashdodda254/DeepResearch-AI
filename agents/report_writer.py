from agents.llm_client import GeminiLLM


class ResearchReportWriter:
    """Generate structured research reports with verifiable references."""

    def __init__(self):
        self.llm = GeminiLLM()

    def generate_references(self, evidence: list) -> str:
        """Generate references using actual retrieved paper metadata."""

        if not evidence:
            return "No verified references available."

        references = []

        for index, item in enumerate(evidence, start=1):
            title = item.get("title", "Unknown title")
            authors = item.get("authors", [])
            published = item.get("published", "Unknown date")
            paper_id = item.get("paper_id", "Unknown ID")
            pdf_url = item.get("pdf_url", "Unavailable")
            page = item.get("page_number", "Unknown")

            if isinstance(authors, list):
                authors = ", ".join(authors)

            reference = (
                f"[Source {index}]\n"
                f"Title: {title}\n"
                f"Authors: {authors or 'Not available'}\n"
                f"Published: {published}\n"
                f"arXiv ID: {paper_id}\n"
                f"PDF: {pdf_url}\n"
                f"Evidence page: {page}\n"
            )

            references.append(reference)

        return "\n".join(references)

    def generate_report(
        self,
        question: str,
        analysis: str,
        evidence: list = None
    ) -> str:
        """Generate report and append verified references."""

        evidence = evidence or []

        system_prompt = """
        You are an academic research report writer.

        Create a professional research report using only
        the supplied research analysis.

        Use this structure:

        1. Title
        2. Abstract
        3. Introduction
        4. Key Findings
        5. Discussion
        6. Limitations
        7. Conclusion

        Important rules:
        - Do not invent research findings.
        - Preserve source citations such as [Source 1].
        - Do not invent authors, dates, statistics or references.
        - Clearly identify gaps in the available evidence.
        - Do not create a References section.
        - Do not introduce unsupported claims.
        """

        prompt = f"""
        RESEARCH QUESTION:
        {question}

        RESEARCH ANALYSIS:
        {analysis}

        Write the final research report using the required structure.
        """

        report = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )

        references = self.generate_references(evidence)

        return (
            report.strip()
            + "\n\n8. References\n"
            + references
        )


if __name__ == "__main__":

    writer = ResearchReportWriter()

    sample_analysis = """
    RAG combines information retrieval with language generation.
    The supplied evidence describes its basic architecture.
    More studies are required to establish its effectiveness.
    """

    sample_evidence = [
        {
            "title": "Example RAG Research Paper",
            "authors": ["Example Author"],
            "published": "2025-01-01",
            "paper_id": "2501.12345",
            "pdf_url": "https://arxiv.org/pdf/2501.12345",
            "page_number": 3
        }
    ]

    report = writer.generate_report(
        "What is Retrieval-Augmented Generation?",
        sample_analysis,
        sample_evidence
    )

    print("\nGENERATED RESEARCH REPORT")
    print("=" * 60)
    print(report)