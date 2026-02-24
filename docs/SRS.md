# Software Requirements Specification (SRS) – Sudharshan

## 1. Purpose
Sudharshan is a RAG-based assistant for analyzing academic PDFs and answering questions grounded in uploaded papers.

## 2. Scope
Upload PDFs → extract/clean → chunk → embed → store FAISS + Postgres → query retrieve → generate answer + citations.

## 3. Functional Requirements
- Upload 1..N PDFs
- Extract + clean + chunk text
- Generate embeddings and index in FAISS
- Persist metadata in PostgreSQL
- Semantic search over chunks
- LLM answer generation using retrieved context
- Return citations (doc title/page/snippet/score)

## 4. Non-Functional Requirements
- Accuracy: grounded answers using only retrieved context
- Latency: <5s typical on CPU for retrieval; generation depends on model
- Security: no file sharing; local persistence
- Portability: Docker-ready (optional)

## 5. Constraints
- CPU-only acceptable; GPU preferred for faster embeddings
- HF Inference requires internet + API token

## 6. Assumptions
- PDFs are extractable (OCR not included by default)
