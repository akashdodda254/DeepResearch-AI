"""
Run retrieval evaluation against the benchmark dataset
and save results as a JSON report.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.researcher import ResearcherAgent
from evaluation.benchmark import BENCHMARK_DATA
from evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


RESULTS_DIR = Path("evaluation/results")


def normalize_paper_id(paper_id):
    """Normalize arXiv IDs by removing version suffixes."""
    return str(paper_id).split("v")[0].strip()


def evaluate_retrieval():
    if not BENCHMARK_DATA:
        raise ValueError("Benchmark dataset is empty.")

    researcher = ResearcherAgent()
    evaluation_results = []

    for index, benchmark in enumerate(BENCHMARK_DATA[:1], start=1):

        question = benchmark["question"]
        relevant_ids = benchmark["relevant_paper_ids"]

        print("\n" + "=" * 65)
        print(f"BENCHMARK QUESTION {index}")
        print("=" * 65)
        print(question)

        retrieved = researcher.research(
            question,
            max_results=3,
            retrieval_top_k=10,
            rerank_top_k=10,
        )

        # Convert retrieved chunks into unique paper IDs.
        retrieved_paper_ids = []
        seen_ids = set()

        for result in retrieved:
            paper_id = normalize_paper_id(
                result.get("paper_id", "")
            )

            if paper_id and paper_id not in seen_ids:
                retrieved_paper_ids.append(paper_id)
                seen_ids.add(paper_id)

        recall = recall_at_k(
            retrieved_paper_ids,
            relevant_ids,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved_paper_ids,
            relevant_ids,
        )

        result = {
            "question": question,
            "relevant_paper_ids": relevant_ids,
            "retrieved_paper_ids": retrieved_paper_ids,
            "recall_at_5": recall,
            "reciprocal_rank": rr,
        }

        evaluation_results.append(result)

        print("\nRetrieved papers:")
        for rank, paper_id in enumerate(retrieved_paper_ids, start=1):
            print(f"{rank}. {paper_id}")

        print(f"\nRecall@5: {recall:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")

    mrr = mean_reciprocal_rank(evaluation_results)

    average_recall = sum(
        result["recall_at_5"]
        for result in evaluation_results
    ) / len(evaluation_results)

    final_results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_questions": len(evaluation_results),
        "average_recall_at_5": average_recall,
        "mean_reciprocal_rank": mrr,
        "results": evaluation_results,
    }

    # Save evaluation results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 65)
    print(f"Benchmark questions: {len(evaluation_results)}")
    print(f"Average Recall@5: {average_recall:.4f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_retrieval()