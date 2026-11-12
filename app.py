import streamlit as st


st.set_page_config(
    page_title="DeepResearch-AI",
    page_icon="🔬",
    layout="wide"
)


@st.cache_resource
def load_research_workflow():
    from graph.research_workflow import research_app
    return research_app


st.title("🔬 DeepResearch-AI")
st.subheader("An Evidence-Grounded Multi-Agent AI Research Assistant")

st.markdown(
    """
    Enter a research question to retrieve academic evidence,
    analyze the findings, and generate a research report.
    """
)

question = st.text_area(
    "Research Question",
    placeholder="Example: How is Retrieval-Augmented Generation used in healthcare?",
    height=120
)

use_mock = st.checkbox(
    "Use mock data (no Gemini API calls)",
    value=True
)

run_button = st.button(
    "Start Research",
    type="primary"
)
if run_button:

    if not question.strip():
        st.warning("Please enter a research question.")

    else:
        try:
            with st.spinner(
                "Researching... This may take a few moments."
            ):
               initial_state = {
                "question": question.strip(),
                "plan": [],
                "evidence": [],
                "analysis": "",
                "report": ""
            }

            if use_mock:
                from mock_workflow import mock_invoke
                result = mock_invoke(initial_state)

            else:
                research_app = load_research_workflow()
                result = research_app.invoke(initial_state)

            st.success("Research completed successfully!")

            st.header("Research Report")

            st.markdown(
                """
                <style>
                div[data-testid="stMarkdownContainer"] p,
                div[data-testid="stMarkdownContainer"] li {
                    font-size: 14px !important;
                    line-height: 1.7 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(result["report"])
            
            st.download_button(
                label="Download Research Report",
                data=result["report"],
                file_name="deepresearch_report.md",
                mime="text/markdown"
            )

            with st.expander("View Research Plan"):
                for item in result["plan"]:
                    st.write(f"- {item}")

            with st.expander("View Retrieved Evidence"):
                if result["evidence"]:
                    for index, item in enumerate(
                        result["evidence"], start=1
                    ):
                        st.markdown(f"**Source {index}**")
                        st.write(
                            "Title:",
                            item.get("title", "Unknown")
                        )
                        st.write(
                            "Paper ID:",
                            item.get("paper_id", "Unknown")
                        )
                        st.write(
                            "Text:",
                            item.get("text", "No text available")
                        )
                        st.divider()
                else:
                    st.info("No evidence was retrieved.")

            with st.expander("View Research Analysis"):
                st.write(result["analysis"])

        except Exception as error:
            st.error("The research workflow encountered an error.")
            st.exception(error)