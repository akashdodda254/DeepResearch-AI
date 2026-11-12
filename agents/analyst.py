from agents.llm_client import GeminiLLM


class ResearchAnalyst:
    """Analyze retrieved academic evidence."""

    def __init__(self):
        self.llm = GeminiLLM()

    def analyze(self, question: str, evidence: list) -> str:

        if not evidence:
            return "No relevant research evidence was found."

        evidence_sections = []

        for index, item in enumerate(evidence, start=1):
            evidence_text = item.get("text", "")

            evidence_sections.append(
                f"""
SOURCE {index}
Title: {item.get('title', 'Unknown')}
Paper ID: {item.get('paper_id', 'Unknown')}
Page: {item.get('page_number', 'Unknown')}

Content:
{evidence_text}
"""
            )

        evidence_text = "\n".join(evidence_sections)

        system_prompt = """
You are an academic research analyst specializing in evidence-based
scientific research.

Your task is to analyze retrieved academic evidence and answer the
research question accurately.

STRICT RULES:

1. Use only the supplied academic evidence.
2. Read all supplied evidence before forming conclusions.
3. Do not invent facts, statistics, methods, or citations.
4. Extract all relevant findings, including specific methods,
   strategies, mechanisms, and limitations.
5. Do not overlook important details present in the evidence.
6. Cite every evidence-based finding using [Source N].
7. Distinguish direct evidence from interpretation.
8. If the evidence does not answer part of the question,
   explicitly identify the information gap.
9. Never claim that information is missing if it is present
   in the supplied evidence.
10. Avoid repeating the same finding unnecessarily.

STRUCTURE YOUR ANALYSIS:

### 1. Direct Evidence
Identify findings that directly answer the research question.

### 2. Important Findings
Explain the relevant methods, approaches, mechanisms,
or strategies described in the evidence.

### 3. Interpretation
Explain what the findings suggest without presenting
unsupported assumptions as facts.

### 4. Limitations
Identify limitations and unanswered aspects supported
by the supplied evidence.

### 5. Conclusion
Provide a concise answer to the research question
based on the available evidence.
"""

        prompt = f"""
RESEARCH QUESTION:
{question}

RETRIEVED ACADEMIC EVIDENCE:
{evidence_text}

Analyze ALL the supplied sources and produce a structured,
evidence-grounded synthesis.

Pay particular attention to specific strategies and methods
that directly address the research question.
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