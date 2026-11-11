"""
Benchmark dataset for evaluating academic paper retrieval.

Relevance labels are based on manual review of arXiv abstracts.
"""

BENCHMARK_DATA = [
    {
        "question": (
            "How is retrieval augmented generation used "
            "to improve medication information retrieval "
            "for healthcare workers?"
        ),
        "relevant_paper_ids": [
            "2502.15722",
        ],
    },
    {
        "question": (
            "How is retrieval augmented generation applied "
            "to medical and healthcare question answering?"
        ),
        "relevant_paper_ids": [
            "2309.16035",
            "2407.18044",
        ],
    },
    {
        "question": (
            "What approaches are used to build reliable "
            "RAG systems for medical and clinical applications?"
        ),
        "relevant_paper_ids": [
            "2412.16701",
            "2504.14917",
        ],
    },
]