from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from agents.planner import ResearchPlanner
from agents.researcher import ResearcherAgent
from agents.analyst import ResearchAnalyst
from agents.report_writer import ResearchReportWriter


# --------------------------------------------------
# 1. Define workflow state
# --------------------------------------------------

class ResearchState(TypedDict):
    question: str
    plan: List[str]
    evidence: List[dict]
    analysis: str
    report: str


# --------------------------------------------------
# 2. Initialize agents
# --------------------------------------------------

planner = ResearchPlanner()
researcher = ResearcherAgent()
analyst = ResearchAnalyst()
writer = ResearchReportWriter()


# --------------------------------------------------
# 3. Research Planner Node
# --------------------------------------------------

def planning_node(state: ResearchState):

    print("\n[1/4] Creating research plan...")

    plan = planner.create_plan(state["question"])

    if not plan:
        raise ValueError("Research planner returned an empty plan.")

    print(f"Generated {len(plan)} research questions.")

    return {"plan": plan}


# --------------------------------------------------
# 4. Researcher Node
# --------------------------------------------------

def research_node(state: ResearchState):

    print("\n[2/4] Collecting research evidence...")

    # Lightweight mode: use only the first sub-question.
    research_question = state["plan"][0]

    print(f"Research query: {research_question}")

    evidence = researcher.research(
        research_question,
        max_results=1
    )

    if not evidence:
        print("Warning: No research evidence retrieved.")

    # Keep only the top 3 evidence passages.
    evidence = evidence[:3]

    print(f"Collected {len(evidence)} evidence passages.")
    if evidence:
        print("\nRetrieved paper:")
        print("Title:", evidence[0].get("title", "Unknown"))
        print("Paper ID:", evidence[0].get("paper_id", "Unknown"))

    return {"evidence": evidence}


# --------------------------------------------------
# 5. Analyst Node
# --------------------------------------------------

def analysis_node(state: ResearchState):

    print("\n[3/4] Analyzing research evidence...")

    if not state["evidence"]:
        analysis = (
            "No relevant academic evidence was retrieved. "
            "A reliable research analysis cannot be generated."
        )
    else:
        analysis = analyst.analyze(
            state["question"],
            state["evidence"]
        )

    return {"analysis": analysis}


# --------------------------------------------------
# 6. Report Writer Node
# --------------------------------------------------

def report_node(state: ResearchState):
    
    print("\n[4/4] Generating research report...")

    if not state["analysis"]:
        raise ValueError("Research analysis is empty.")

    report = writer.generate_report(
        state["question"],
        state["analysis"],
        state["evidence"]
    )

    return {"report": report}

# --------------------------------------------------
# 7. Build LangGraph workflow
# --------------------------------------------------

workflow = StateGraph(ResearchState)

workflow.add_node("planner", planning_node)
workflow.add_node("researcher", research_node)
workflow.add_node("analyst", analysis_node)
workflow.add_node("report_writer", report_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "report_writer")
workflow.add_edge("report_writer", END)

research_app = workflow.compile()


# --------------------------------------------------
# 8. Run application
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "How is Retrieval-Augmented Generation "
        "used in healthcare?"
    )

    initial_state = {
        "question": question,
        "plan": [],
        "evidence": [],
        "analysis": "",
        "report": ""
    }

    print("\n" + "=" * 60)
    print("       DEEPRESEARCH-AI")
    print(" Evidence-Grounded Research Assistant")
    print("=" * 60)

    try:

        result = research_app.invoke(initial_state)

        print("\n\nFINAL RESEARCH REPORT")
        print("=" * 70)

        print(result["report"])

        print("\n" + "=" * 70)
        print("RESEARCH WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as error:

        print("\nRESEARCH WORKFLOW FAILED")
        print("=" * 70)
        print(f"Error: {error}")
        raise