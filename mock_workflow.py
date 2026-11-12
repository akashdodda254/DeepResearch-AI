def mock_invoke(state):
    
    question = state["question"]

    return {
        "question": question,

        "plan": [
            question,
            "Identify relevant academic findings",
            "Evaluate evidence and limitations"
        ],

        "evidence": [
            {
                "title": "Example Healthcare RAG Research",
                "paper_id": "2502.15722",
                "page_number": 5,
                "text": (
                    "Retrieval-Augmented Generation can improve "
                    "access to relevant medical information. "
                    "However, systems require evidence grounding "
                    "and careful validation."
                )
            },
            {
                "title": "Example Clinical AI Research",
                "paper_id": "2412.16701",
                "page_number": 8,
                "text": (
                    "Uncertainty flagging and human review can "
                    "help identify potentially unreliable outputs."
                )
            },
            {
                "title": "Example Medical AI Evaluation",
                "paper_id": "2407.18044",
                "page_number": 4,
                "text": (
                    "Evaluation of retrieved evidence is important "
                    "for assessing the reliability of medical "
                    "question-answering systems."
                )
            }
        ],

        "analysis": """
        ### Important Findings

        - Evidence grounding supports more reliable answers.
        - Uncertainty flagging helps identify weak evidence.
        - Human review is important for high-stakes applications.

        ### Limitations

        These are demonstration findings, not verified research results.
        """,

        "report": """
# Healthcare AI Research Report

## Abstract

This is a demonstration report generated using mock data.

## Key Findings

1. Evidence grounding can support reliable information retrieval.
2. Uncertainty flagging can identify potentially unreliable outputs.
3. Human review is important in high-stakes applications.

## Conclusion

This sample demonstrates the report display functionality
of DeepResearch-AI.

**Note:** This is mock content, not an actual research result.
"""
    }