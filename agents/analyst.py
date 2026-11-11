from agents.llm_client import GeminiLLM

class ResearchAnalyst:
    """Analyze retrieved academic evidence."""

    def __init__(self):
        self.llm = GeminiLLM()

    def analyze(self, question: str, evidence: list) -> str:

        if not evidence:
            return "No relevant research evidence was found."

        evidence_text = ""

        for index, item in enumerate(evidence[:1], start=1):
            evidence_text += (
                f"\nSOURCE {index}\n"
                f"Title: {item.get('title', 'Unknown')}\n"
                f"Paper ID: {item.get('paper_id', 'Unknown')}\n"
                f"Page: {item.get('page_number', 'Unknown')}\n"
                f"Content: {item.get('text', '')[:300]}\n"
            )

        system_prompt = """
        You are an academic research analyst.

        Analyze the supplied academic evidence to answer the question.

        Rules:
        1. Use only the supplied evidence.
        2. Do not invent facts, statistics, or citations.
        3. Identify important findings and limitations.
        4. If the evidence does not answer the question, say so.
        5. Cite sources using [Source 1], [Source 2], etc.
        6. Distinguish direct evidence from interpretation.
        """

        prompt = f"""
        RESEARCH QUESTION:
        {question}

        RETRIEVED ACADEMIC EVIDENCE:
        {evidence_text}

        Analyze the evidence and provide a structured synthesis.
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )


if __name__ == "__main__":
    analyst = ResearchAnalyst()

    sample_evidence = [
        {
            "title": "Example RAG Research Paper",
            "paper_id": "example-001",
            "page_number": 3,
            "text": (
                "Retrieval-Augmented Generation combines "
                "retrieval mechanisms with language generation."
            )
        }
    ]

    result = analyst.analyze(
        "What is Retrieval-Augmented Generation?",
        sample_evidence
    )

    print("\nRESEARCH ANALYSIS")
    print("=" * 50)
    print(result)