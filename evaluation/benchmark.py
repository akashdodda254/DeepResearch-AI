"""
Benchmark dataset for evaluating academic paper retrieval.

Each benchmark question contains a list of manually identified
relevant arXiv paper IDs.
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
            "to medical question answering?"
        ),
        "relevant_paper_ids": [
            "2309.16035",
            "2407.18044",
        ],
    },
    {
        "question": (
            "How can retrieval augmented generation support "
            "clinical applications using medical literature?"
        ),
        "relevant_paper_ids": [
            "2412.16701",
            "2504.14917",
        ],
    },
]