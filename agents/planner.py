class ResearchPlanner:
    """Create a lightweight research plan."""

    def create_plan(self, research_question: str) -> list:

        if not research_question.strip():
            raise ValueError("Research question cannot be empty.")

        return [
            research_question.strip()
        ]


if __name__ == "__main__":

    planner = ResearchPlanner()

    question = "How is Retrieval-Augmented Generation used in healthcare?"

    print("Creating research plan...\n")

    plan = planner.create_plan(question)

    for number, sub_question in enumerate(plan, start=1):
        print(f"{number}. {sub_question}")

    print("\nResearch plan created successfully!")