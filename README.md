# Chat with energy document (RAG-Based)

**Tech Stack:** FastAPI (backend), PostgreSQL (SQLAlchemy), FAISS (vector store), sentence-transformers (embeddings), Hugging Face Inference (Mixtral 8x7B) or local fallback, Streamlit (frontend), pdfplumber/PyPDF2 (PDF), Git/GitHub.

## Features
- Upload 1..N PDFs
- Auto extract, clean, chunk (500–800 tokens approx), embed with `all-MiniLM-L6-v2`
- Store embeddings in FAISS locally + metadata in PostgreSQL
- Query with semantic retrieval + LLM generation (grounded answers with citations)
- Streamlit UI showing upload, indexing progress, and sources

## Quickstart

### 1) Create and fill your `.env`
```bash
cp .env.example .env
# edit: DATABASE_URL, HF_API_TOKEN (if using Hugging Face Inference), etc.
```

### 2) Create DB and tables
```bash
python scripts/init_db.py
```

### 3) Run backend
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
# Docs at: http://localhost:8000/docs
```

### 4) Run frontend
```bash
streamlit run frontend/app.py
```

### Notes
- FAISS index + pickle metadata are saved in `./data`.
- Hugging Face **Mixtral** via Inference API needs `HF_API_TOKEN` and model name in `.env`.
- If you don't have Inference credits, the code falls back to a local small seq2seq model (`flan-t5-base`) for demo. You can change it.
- On first run, `sentence-transformers` will download the embedding model (~90MB).

## Project Structure
```
sudharshan/
  backend/
    app.py            # FastAPI API
    rag.py            # RAG pipeline: chunk, embed, index, retrieve, generate
    db.py             # SQLAlchemy setup + session
    models.py         # ORM models
    settings.py       # Env + config
    utils/
      __init__.py
      pdf_utils.py    # PDF extraction/cleaning
  frontend/
    app.py            # Streamlit UI
  scripts/
    init_db.py        # Create tables
  data/               # FAISS + metadata
  docs/
    SDS.md            # Software Design Spec (skeleton)
    SRS.md            # Software Requirements Spec (skeleton)
  requirements.txt
  .env.example
  README.md
```

## API (selected)
- `POST /upload` : upload and index PDFs
- `GET /docs` : list known PDFs
- `POST /query` : ask a question -> returns answer + sources

## Citations
Each answer returns a `sources` array: `[{"doc_title": "...","page": 3,"score": 0.79,"snippet": "..."}]`

## Deployment
- Hugging Face Space (Streamlit) or Render (backend+frontend). For Spaces, prefer uploads < 100MB and keep CPU-only. For Render, add a managed PostgreSQL and persistent disk for FAISS.

## License
MIT
