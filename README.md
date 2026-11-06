# DeepResearch-AI
### An Evidence-Grounded Multi-Agent AI Research Assistant

<p align="center">
  <b>Automating academic research through multi-agent orchestration, hybrid retrieval, and evidence-grounded reasoning.</b>
</p>

---

## Overview

DeepResearch-AI is an AI-powered research assistant designed to automate the academic research process.

The system aims to transform complex research questions into structured research reports by combining multi-agent orchestration, Retrieval-Augmented Generation (RAG), academic paper retrieval, and evidence-based reasoning.

Instead of generating answers from language models alone, DeepResearch-AI retrieves relevant academic literature, analyzes supporting evidence, and generates responses grounded in research documents.

## Key Features

- **Multi-Agent Research Workflow:** Specialized agents for research planning, question decomposition, information retrieval, analysis, verification, and report generation.
- **Hybrid Retrieval-Augmented Generation:** Combines FAISS-based semantic search with BM25 keyword retrieval.
- **Cross-Encoder Reranking:** Improves retrieval relevance by reranking candidate passages.
- **Academic Paper Retrieval:** Integrates the arXiv API to discover relevant research papers.
- **PDF Processing:** Extracts academic paper content and metadata using PyMuPDF.
- **Citation-Grounded Responses:** Connects generated findings to their supporting research sources.
- **Iterative Research:** Supports additional retrieval when evidence is insufficient.
- **Research Evaluation:** Evaluates retrieval quality and response reliability using information retrieval metrics.

## System Architecture

```text
                 USER QUERY
                      |
                      v
               RESEARCH PLANNER
                      |
                      v
             QUESTION DECOMPOSER
                      |
                      v
                RESEARCH AGENT
                      |
                      v
                 ARXIV API
                      |
                      v
             PAPER DOWNLOAD
                      |
                      v
              PDF PROCESSING
                  PyMuPDF
                      |
                      v
              DOCUMENT CHUNKING
                      |
                      v
                HYBRID RAG
                 /       \
                /         \
             FAISS        BM25
                \         /
                 \       /
              CROSS-ENCODER
                 RERANKING
                      |
                      v
               ANALYSIS AGENT
                      |
                      v
             VERIFICATION AGENT
                      |
                      v
                REPORT AGENT
                      |
                      v
             RESEARCH REPORT
```

LangGraph will orchestrate the agents and manage the research workflow.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| LangGraph | Multi-agent orchestration |
| Ollama | Local language model inference |
| FAISS | Vector similarity search |
| BM25 | Keyword-based retrieval |
| Sentence Transformers | Text embeddings |
| Cross-Encoder | Document reranking |
| arXiv API | Academic paper retrieval |
| PyMuPDF | PDF text extraction |
| Streamlit | User interface |
| Pydantic | Data validation |

## Research Evaluation

The system will be evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Recall@5 | Measures relevant documents retrieved in the top five results |
| Mean Reciprocal Rank (MRR) | Measures the ranking quality of relevant results |
| Citation Precision | Measures whether generated citations correctly support claims |
| Answer Faithfulness | Measures whether generated answers remain grounded in retrieved evidence |

## Project Structure

```text
DeepResearch-AI/
│
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   ├── analyst.py
│   ├── verifier.py
│   └── report_generator.py
│
├── retrieval/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── reranker.py
│
├── ingestion/
│   ├── arxiv_loader.py
│   ├── pdf_processor.py
│   └── chunking.py
│
├── evaluation/
│   ├── retrieval_metrics.py
│   └── response_metrics.py
│
├── graph/
│   ├── state.py
│   └── workflow.py
│
├── data/
├── reports/
│
├── app.py
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- Ollama (for local LLM inference)

### Installation

Clone the repository:

```bash
git clone https://github.com/akashdodda254/DeepResearch-AI.git
cd DeepResearch-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

*Detailed installation and execution instructions will be added as development progresses.*

## Development Roadmap

- [ ] Project architecture and environment setup
- [ ] arXiv academic paper ingestion
- [ ] PDF extraction and document processing
- [ ] Document chunking and metadata management
- [ ] FAISS vector retrieval
- [ ] BM25 keyword retrieval
- [ ] Hybrid retrieval pipeline
- [ ] Cross-encoder reranking
- [ ] Multi-agent implementation using LangGraph
- [ ] Citation verification and evidence analysis
- [ ] Research report generation
- [ ] Retrieval and response evaluation
- [ ] Streamlit application
- [ ] Testing and documentation

## Project Status

**Currently under active development.**

This project is being developed incrementally, with individual components tested before integration into the complete multi-agent research workflow.

## Author

**Akash Dodda**

GitHub: [@akashdodda254](https://github.com/akashdodda254)

---

If you find this project interesting, consider giving it a star!