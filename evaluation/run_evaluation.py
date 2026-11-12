"""
Evaluate individual retrieval stages and save results as JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.researcher import ResearcherAgent
from evaluation.benchmark import BENCHMARK_DATA

from evaluation.retrieval_metrics import (
    reciprocal_rank,
    mean_reciprocal_rank,
)

from evaluation.stage_metrics import (
    candidate_recall,
    stage_recall_at_k,
    unique_paper_ids,
)


RESULTS_DIR = Path("evaluation/results")


def evaluate_retrieval():

    if not BENCHMARK_DATA:
        raise ValueError("Benchmark dataset is empty.")

    researcher = ResearcherAgent()
    evaluation_results = []

    for index, benchmark in enumerate(BENCHMARK_DATA, start=1):

        question = benchmark["question"]
        relevant_ids = benchmark["relevant_paper_ids"]

        print("\n" + "=" * 65)
        print(f"BENCHMARK QUESTION {index}")
        print("=" * 65)
        print(question)

        stages = researcher.research(
            question,
            max_results=5,
            retrieval_top_k=15,
            rerank_top_k=15,
            return_stages=True,
        )

        if not isinstance(stages, dict):
            print("No retrieval results found.")
            stages = {
                "candidate_paper_ids": [],
                "hybrid_results": [],
                "reranked_results": [],
            }

        # Stage 1: arXiv candidate retrieval
        candidate_ids = stages["candidate_paper_ids"]

        candidate_score = candidate_recall(
            candidate_ids,
            relevant_ids,
        )

        # Stage 2: Hybrid retrieval
        hybrid_results = stages["hybrid_results"]

        hybrid_score = stage_recall_at_k(
            hybrid_results,
            relevant_ids,
            k=5,
        )

        # Stage 3: Cross-encoder reranking
        reranked_results = stages["reranked_results"]

        reranked_ids = unique_paper_ids(reranked_results)

        reranked_score = stage_recall_at_k(
            reranked_results,
            relevant_ids,
            k=5,
        )

        rr = reciprocal_rank(
            reranked_ids,
            relevant_ids,
        )

        result = {
            "question": question,
            "relevant_paper_ids": relevant_ids,

            "candidate_paper_ids": candidate_ids,
            "candidate_recall": candidate_score,

            "hybrid_retrieved_paper_ids": unique_paper_ids(
                hybrid_results
            ),
            "hybrid_recall_at_5": hybrid_score,

            "reranked_paper_ids": reranked_ids,
            "reranked_recall_at_5": reranked_score,

            "reciprocal_rank": rr,
        }

        evaluation_results.append(result)

        print("\nSTAGE EVALUATION")
        print(f"arXiv Candidate Recall: {candidate_score:.4f}")
        print(f"Hybrid Recall@5: {hybrid_score:.4f}")
        print(f"Reranked Recall@5: {reranked_score:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")

    # Calculate average metrics
    average_candidate_recall = sum(
        item["candidate_recall"]
        for item in evaluation_results
    ) / len(evaluation_results)

    average_hybrid_recall = sum(
        item["hybrid_recall_at_5"]
        for item in evaluation_results
    ) / len(evaluation_results)

    average_reranked_recall = sum(
        item["reranked_recall_at_5"]
        for item in evaluation_results
    ) / len(evaluation_results)

    mrr = mean_reciprocal_rank([
        {
            "retrieved_paper_ids": item["reranked_paper_ids"],
            "relevant_paper_ids": item["relevant_paper_ids"],
        }
        for item in evaluation_results
    ])

    final_results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_questions": len(evaluation_results),

        "average_candidate_recall": average_candidate_recall,
        "average_hybrid_recall_at_5": average_hybrid_recall,
        "average_reranked_recall_at_5": average_reranked_recall,

        "mean_reciprocal_rank": mrr,

        "results": evaluation_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "stage_evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL STAGE-LEVEL EVALUATION")
    print("=" * 65)
    print(f"Benchmark questions: {len(evaluation_results)}")
    print(f"Candidate Recall: {average_candidate_recall:.4f}")
    print(f"Hybrid Recall@5: {average_hybrid_recall:.4f}")
    print(f"Reranked Recall@5: {average_reranked_recall:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_retrieval()