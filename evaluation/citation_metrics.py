"""
Basic citation validation for generated research reports.

Checks whether report citations refer to available evidence sources.
This does not determine whether a cited source actually supports a claim.
"""

import re


def extract_citations(report: str) -> list:
    """Extract citation labels such as [Source 1] from a report."""
    return re.findall(r"\[Source\s+\d+\]", report)


def citation_precision(report: str, evidence: list) -> float:
    """
    Calculate the proportion of report citations that resolve
    to available evidence records.
    """

    citations = extract_citations(report)

    if not citations:
        return 0.0

    available_sources = {
        f"[Source {index}]"
        for index in range(1, len(evidence) + 1)
    }

    valid_citations = [
        citation
        for citation in citations
        if citation in available_sources
    ]

    return len(valid_citations) / len(citations)


def validate_citations(report: str, evidence: list) -> dict:
    """Return a summary of citation validation."""

    citations = extract_citations(report)

    available_sources = {
        f"[Source {index}]"
        for index in range(1, len(evidence) + 1)
    }

    valid = [c for c in citations if c in available_sources]
    invalid = [c for c in citations if c not in available_sources]

    return {
        "total_citations": len(citations),
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "citation_precision": citation_precision(report, evidence),
        "invalid_references": invalid,
    }