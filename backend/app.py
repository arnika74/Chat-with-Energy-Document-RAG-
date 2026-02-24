import os, uuid, tempfile
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import Document, Chunk
from .settings import INDEX_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from .utils.pdf_utils import extract_pdf_text_per_page
from .rag import chunk_text, upsert_embeddings, search, generate_answer

app = FastAPI(title="Sudharshan RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure tables exist (for dev convenience)
Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/docs-list")
def list_docs(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    out = [{"id": d.id, "title": d.title, "pages": d.pages, "created_at": d.created_at.isoformat()} for d in docs]
    return {"documents": out}

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    os.makedirs(INDEX_DIR, exist_ok=True)
    added_total = 0
    doc_infos = []

    for f in files:
        suffix = os.path.splitext(f.filename)[-1].lower()
        if suffix != ".pdf":
            return JSONResponse(status_code=400, content={"error": f"Only PDF supported. Got {suffix} for {f.filename}"})
        # Save temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await f.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text per page
        pages = extract_pdf_text_per_page(tmp_path)

        # Save doc record
        doc = Document(title=f.filename, path=tmp_path, pages=len(pages))
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Build chunks
        rows = []
        for page_no, text in pages:
            if not text.strip():
                continue
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for ch in chunks:
                rows.append({
                    "doc_id": doc.id,
                    "doc_title": doc.title,
                    "page": page_no,
                    "text": ch
                })
        
        # Upsert to FAISS + write chunk rows to DB
        added_ids = upsert_embeddings(rows)
        for vec_id, row in zip(added_ids, rows):
            c = Chunk(doc_id=row["doc_id"], page=row["page"], text=row["text"], vec_id=vec_id)
            db.add(c)
        db.commit()

        added_total += len(rows)
        doc_infos.append({"doc_id": doc.id, "title": doc.title, "pages": doc.pages, "chunks": len(rows)})

    return {"indexed": added_total, "docs": doc_infos}

@app.post("/query")
async def query_api(q: str = Form(...), top_k: int = Form(TOP_K)):
    hits = search(q, k=top_k)
    answer = generate_answer(q, hits)
    sources = []
    for h in hits:
        sources.append({
            "doc_title": h.get("doc_title"),
            "page": h.get("page"),
            "score": h.get("score", 0.0),
            "snippet": h.get("text", "")[:300]
        })
    return {"answer": answer, "sources": sources}
