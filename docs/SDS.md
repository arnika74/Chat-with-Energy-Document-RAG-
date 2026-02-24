# Software Design Specification (SDS) – Sudharshan

## Architecture
- REST API (FastAPI)
- DB (PostgreSQL via SQLAlchemy)
- Vector Store (FAISS)
- Embeddings (sentence-transformers)
- LLM Generation (HF Inference Mixtral) with local fallback

## Data Model
- Document(id, title, path, pages, created_at)
- Chunk(id, doc_id, page, text, embedding_vector_id, created_at)

## RAG Flow
1) `/upload`: extract text per page → clean → chunk with overlap → embed → add to FAISS → upsert metadata to Postgres
2) `/query`: embed query → FAISS top-k → prompt → generate → return with citations

## Key Modules
- `utils/pdf_utils.py` for stable extraction
- `rag.py` for chunking, embedding, FAISS index, retrieval, generation
- `db.py` & `models.py` for persistence & metadata
