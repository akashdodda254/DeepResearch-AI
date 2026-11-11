import unittest

from evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)
from evaluation.citation_metrics import (
    extract_citations,
    citation_precision,
    validate_citations,
)
from evaluation.faithfulness import create_faithfulness_review


class TestRetrievalMetrics(unittest.TestCase):

    def test_recall_at_k(self):
        retrieved = ["paper1", "paper2", "paper3"]
        relevant = ["paper2", "paper3", "paper4"]

        score = recall_at_k(retrieved, relevant, k=2)

        self.assertAlmostEqual(score, 1 / 3)

    def test_reciprocal_rank(self):
        retrieved = ["paper1", "paper2", "paper3"]
        relevant = ["paper3"]

        self.assertAlmostEqual(
            reciprocal_rank(retrieved, relevant),
            1 / 3
        )

    def test_mrr(self):
        results = [
            {
                "retrieved_paper_ids": ["paper1", "paper2"],
                "relevant_paper_ids": ["paper2"],
            },
            {
                "retrieved_paper_ids": ["paper3", "paper4"],
                "relevant_paper_ids": ["paper3"],
            },
        ]

        self.assertAlmostEqual(
            mean_reciprocal_rank(results),
            0.75
        )


class TestCitationMetrics(unittest.TestCase):

    def test_extract_citations(self):
        report = "Finding one [Source 1]. Finding two [Source 2]."

        citations = extract_citations(report)

        self.assertEqual(
            citations,
            ["[Source 1]", "[Source 2]"]
        )

    def test_citation_precision(self):
        report = "Finding one [Source 1]. Finding two [Source 3]."
        evidence = [{"title": "Paper One"}, {"title": "Paper Two"}]

        self.assertAlmostEqual(
            citation_precision(report, evidence),
            0.5
        )

    def test_validate_citations(self):
        report = "Finding [Source 1] and [Source 4]."
        evidence = [{"title": "Paper One"}]

        result = validate_citations(report, evidence)

        self.assertEqual(result["valid_citations"], 1)
        self.assertEqual(result["invalid_citations"], 1)


class TestFaithfulnessReview(unittest.TestCase):

    def test_create_review(self):
        report = "RAG can support medical information retrieval."

        evidence = [{
            "title": "Example Paper",
            "paper_id": "1234.5678",
            "page_number": 2,
            "text": "Example evidence passage."
        }]

        result = create_faithfulness_review(report, evidence)

        self.assertEqual(
            result["review_status"],
            "pending_manual_review"
        )
        self.assertEqual(len(result["evidence"]), 1)


if __name__ == "__main__":
    unittest.main()