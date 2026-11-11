"""
Metrics for evaluating academic paper retrieval.

Includes:
- Recall@K
- Reciprocal Rank
- Mean Reciprocal Rank (MRR)
"""


def normalize_paper_id(paper_id: str) -> str:
    """Remove arXiv version suffixes for consistent ID comparison."""
    return str(paper_id).split("v")[0].strip()


def recall_at_k(
    retrieved_paper_ids: list,
    relevant_paper_ids: list,
    k: int = 5
) -> float:
    """
    Calculate Recall@K.

    Recall@K = relevant papers retrieved in top K
               / total relevant papers
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_paper_ids:
        raise ValueError("Relevant paper IDs cannot be empty.")

    retrieved = {
        normalize_paper_id(pid)
        for pid in retrieved_paper_ids[:k]
    }

    relevant = {
        normalize_paper_id(pid)
        for pid in relevant_paper_ids
    }

    return len(retrieved.intersection(relevant)) / len(relevant)


def reciprocal_rank(
    retrieved_paper_ids: list,
    relevant_paper_ids: list
) -> float:
    """
    Calculate Reciprocal Rank.

    Returns the reciprocal of the rank of the first
    relevant paper. Returns 0 if none is retrieved.
    """

    relevant = {
        normalize_paper_id(pid)
        for pid in relevant_paper_ids
    }

    if not relevant:
        raise ValueError("Relevant paper IDs cannot be empty.")

    for rank, paper_id in enumerate(retrieved_paper_ids, start=1):
        if normalize_paper_id(paper_id) in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(results: list) -> float:
    """
    Calculate MRR across multiple benchmark questions.

    Each result must contain:
    - retrieved_paper_ids
    - relevant_paper_ids
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    scores = []

    for result in results:
        score = reciprocal_rank(
            result["retrieved_paper_ids"],
            result["relevant_paper_ids"]
        )
        scores.append(score)

    return sum(scores) / len(scores)