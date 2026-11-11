from agents.report_writer import ResearchReportWriter


def main():

    writer = ResearchReportWriter()

    sample_analysis = """
    RAG combines information retrieval with language generation.
    The supplied evidence describes its use in healthcare
    medication information retrieval.
    """

    sample_evidence = [
        {
            "title": "RAG for Healthcare Medication Retrieval",
            "authors": ["Example Author"],
            "published": "2025-01-28",
            "paper_id": "2502.15722v1",
            "pdf_url": "https://arxiv.org/pdf/2502.15722v1",
            "page_number": 2
        }
    ]

    print("Testing Gemini report generation...")

    report = writer.generate_report(
        question="How is RAG used in healthcare?",
        analysis=sample_analysis,
        evidence=sample_evidence
    )

    print("\nGENERATED RESEARCH REPORT")
    print("=" * 60)
    print(report)

    print("\nReport generation successful!")


if __name__ == "__main__":
    main()