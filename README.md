````markdown
# AI-Powered Resume Ranking System

A lightweight, local-first tool that helps compare multiple resumes against a job description.  
It uses a blend of LLM extraction, semantic similarity, keyword matching, and skill-based scoring to rank candidates in a clear and explainable way.

---

## Features

- Upload one job description and multiple resumes (`PDF` or `TXT`).
- Local text extraction and structured analysis using an LLM running through **Ollama**.
- Hybrid scoring model combining:
  - semantic embeddings (BGE + FAISS),
  - keyword search (BM25),
  - skill match quality,
  - experience, recency, projects, and other simple signals.
- Generates an overall fit score with a short explanation.
- Clean Streamlit interface for viewing results and downloading outputs.

---

## How It Works (Brief)

1. Extracts and structures content from the JD and resumes using a **local LLM**.
2. Normalizes and matches skills.
3. Uses embeddings + BM25 to measure resume relevance.
4. Combines all scores into a final weighted ranking.
5. Displays results with optional visualizations.

Everything runs **fully on your machine**—no cloud services required.

---

## Installation

```powershell
pip install streamlit sentence-transformers faiss-cpu rank-bm25 pypdf httpx pandas numpy plotly tqdm
````

Install and run Ollama (local LLM runtime):

```powershell
ollama pull llama3.2:3b-instruct-q4_K_M
```

---

## Run the App

```powershell
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Usage

1. Upload a job description.
2. Upload one or more resumes.
3. Click **Run Candidate Ranking**.
4. View ranked results and brief explanations.

---

## Notes

* The system depends on a **local LLM** running via Ollama.
* Some scoring elements use simple heuristics and may need tuning depending on the role.
* For many resumes, enabling caching is helpful.

---


