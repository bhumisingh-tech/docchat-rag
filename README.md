# DocChat — Local Document Q&A with RAG

A runnable Streamlit app that answers questions from PDF, DOCX, and TXT documents. Uses TF-IDF retrieval and an optional local Ollama language model. No API key required.

## Requirements
- Python 3.10–3.12 recommended
- Optional: Ollama for AI-generated answers (otherwise use Evidence-only mode)

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local address Streamlit prints (usually http://localhost:8501).

## Enable AI answers (optional)

1. Install Ollama: https://ollama.com/download
2. In another terminal run `ollama pull llama3.2:3b`.
3. Start Ollama if it is not running (`ollama serve`).
4. Choose **Local AI (Ollama)** in the app sidebar.

If your machine cannot run the model, select **Evidence only (no AI required)**; this retrieves and displays relevant source passages without generation.

## How RAG works
1. Extract text from uploaded documents (PDF pages retain page numbers).
2. Split text into overlapping ~900-character chunks.
3. Vectorize chunks using TF-IDF (unigrams and bigrams).
4. Compare the question to chunks with cosine similarity.
5. Send top chunks and the question to Ollama with a strict grounding prompt, or display chunks directly in Evidence-only mode.
6. Show sources and similarity scores so answers can be checked.

## Limitations
- Scanned/image-only PDFs need OCR; this version only reads embedded text.
- TF-IDF is lexical retrieval, not embedding-based semantic retrieval.
- An LLM can still make mistakes; verify the cited evidence.
- No data is sent to cloud AI services by this app. Ollama runs locally unless you configure a remote Ollama URL.

## Project structure
- `app.py`: Streamlit UI, extraction, chunking, retrieval, local AI generation
- `requirements.txt`: Python dependencies
- `tests/test_core.py`: basic automated tests
- `sample_notes.txt`: demo document

## Viva questions
**What is RAG?** Retrieve relevant document passages and use them as context for an answer.

**Why chunk?** Small overlapping passages improve retrieval and fit the model's context window.

**Why TF-IDF?** It is transparent, fast, and runs offline without a paid API or embedding model.

**What is cosine similarity?** A measure of directional similarity between document and question vectors.

**How do you reduce hallucinations?** Restrict generation to retrieved passages, request source citations, and show evidence.

**Why Ollama?** It serves a local language model through an HTTP API without an API key.

## Free public deployment (Streamlit Community Cloud)
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `sample_notes.txt` from this folder.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository, branch `main`, and entrypoint `app.py`.
4. Deploy and open the resulting `*.streamlit.app` link.
5. Select **Evidence-based (free cloud demo)**; upload `sample_notes.txt` and ask questions.

**Important:** Evidence-based mode retrieves passages; it does not synthesize AI answers. Ollama requires a separately reachable server, so the local AI mode will not work on Streamlit Cloud using localhost. Do not upload sensitive documents to a public demo. Free hosting may sleep or have resource limits.
