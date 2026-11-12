"""
Stage-level retrieval evaluation metrics.
"""


def normalize_paper_id(paper_id):
    """Remove arXiv version suffixes."""
    return str(paper_id).split("v")[0].strip()


def unique_paper_ids(documents):
    """Extract unique paper IDs while preserving ranking order."""

    paper_ids = []
    seen = set()

    for document in documents:
        paper_id = normalize_paper_id(
            document.get("paper_id", "")
        )

        if paper_id and paper_id not in seen:
            paper_ids.append(paper_id)
            seen.add(paper_id)

    return paper_ids


def candidate_recall(candidate_paper_ids, relevant_paper_ids):
    """
    Measure the proportion of relevant papers found
    during initial arXiv candidate search.
    """

    if not relevant_paper_ids:
        raise ValueError("Relevant paper IDs cannot be empty.")

    candidates = {
        normalize_paper_id(pid)
        for pid in candidate_paper_ids
    }

    relevant = {
        normalize_paper_id(pid)
        for pid in relevant_paper_ids
    }

    return len(candidates.intersection(relevant)) / len(relevant)


def stage_recall_at_k(documents, relevant_paper_ids, k=5):
    """
    Calculate paper-level Recall@K for a retrieval stage.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_paper_ids:
        raise ValueError("Relevant paper IDs cannot be empty.")

    retrieved_ids = unique_paper_ids(documents)[:k]

    retrieved = set(retrieved_ids)

    relevant = {
        normalize_paper_id(pid)
        for pid in relevant_paper_ids
    }

    return len(retrieved.intersection(relevant)) / len(relevant)