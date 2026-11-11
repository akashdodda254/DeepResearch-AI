"""
Faithfulness evaluation utilities.

Provides a manual review structure for checking whether
research report claims are supported by retrieved evidence.

This module does not automatically judge factual correctness.
"""


def create_faithfulness_review(report: str, evidence: list) -> dict:
    """
    Prepare report claims and evidence for manual review.

    Each claim must be reviewed by a human.
    """

    if not report.strip():
        raise ValueError("Report cannot be empty.")

    evidence_summary = []

    for index, item in enumerate(evidence, start=1):
        evidence_summary.append({
            "source": f"[Source {index}]",
            "title": item.get("title", "Unknown title"),
            "paper_id": item.get("paper_id", "Unknown ID"),
            "page_number": item.get("page_number", "Unknown"),
            "text": item.get("text", ""),
        })

    return {
        "report": report,
        "evidence": evidence_summary,
        "review_status": "pending_manual_review",
        "instructions": [
            "Identify individual factual claims in the report.",
            "Locate the evidence cited for each claim.",
            "Mark each claim as supported, partially supported, "
            "unsupported, or unverifiable.",
            "Record the evidence passage supporting your decision.",
        ],
    }