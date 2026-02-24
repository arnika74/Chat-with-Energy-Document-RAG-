import os, pickle, time, math, re, json, requests
from typing import List, Dict, Tuple
import numpy as np
import faiss
from google import genai
# import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from .settings import (
    INDEX_DIR,
    EMBED_MODEL,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HF_API_TOKEN,
    HF_GENERATION_MODEL,
    FALLBACK_GENERATION_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from sentence_transformers import CrossEncoder
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ---------- Embeddings ----------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype('float32')

# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Approximate by characters (safe + fast). Token-based can be added with tiktoken if desired.
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
        if start == end:
            break
    return [c for c in chunks if c.strip()]

# ---------- Index Files ----------
FAISS_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH  = os.path.join(INDEX_DIR, "meta.pkl")

def load_index():
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(384)  # all-MiniLM-L6-v2 -> 384 dims
        meta = {"rows": [], "next_row": 0}
    return index, meta

def save_index(index, meta):
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

# ---------- Retrieval ----------
def search(query: str, k: int = TOP_K):
    index, meta = load_index()
    if index.ntotal == 0:
        return []

    # Over-retrieve e.g. 3x then rerank
    over_k = min(index.ntotal, max(k*3, k))
    qv = embed_texts([query])
    D, I = index.search(qv, over_k)

    cand_rows = []
    for score, row in zip(D[0], I[0]):
        if row == -1 or row >= len(meta["rows"]):
            continue
        r = meta["rows"][row].copy()
        r["score"] = float(score)  # FAISS similarity (cosine since normalized)
        cand_rows.append(r)

    if not cand_rows:
        return []

    # Rerank with CrossEncoder (query, text) relevance
    reranker = get_reranker()
    pairs = [(query, r["text"]) for r in cand_rows]
    rel_scores = reranker.predict(pairs)
    for r, rs in zip(cand_rows, rel_scores):
        r["rerank_score"] = float(rs)

    # Sort by rerank then faiss score as tiebreaker
    cand_rows.sort(key=lambda x: (x["rerank_score"], x["score"]), reverse=True)
    return cand_rows[:k]

# ---------- Generation ----------
# ---------- Generation (Gemini) ----------
SYS_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer the user's question clearly and directly using the provided context. "
    "Do not mention document names, page numbers, or citations. "
    "Do not refer to the context explicitly. "
    "If the answer is not in the context, say you don't know."
)




def local_fallback_generate(prompt: str) -> str:
    """
    Very lightweight fallback using simple extractive approach.
    Strategy: return top context bullets (first 3 chunks) and a short synthesis line.
    """
    lines = prompt.split("\n")
    ctx = [ln for ln in lines if ln.startswith("CTX")]
    ctx = ctx[:3]
    synthesis = (
        "Based on the retrieved context, the key points above address the query. "
        "For more detailed understanding, please refer to the cited pages."
    )
    return "\n".join(ctx + ["", synthesis])


def build_prompt(query: str, retrieved: List[Dict]) -> str:
    ctx_lines = []
    for r in retrieved:
        snippet = r.get("text", "")[:300]
        ctx_lines.append(snippet)
        prompt = (
            f"{SYS_PROMPT}\n\n"
            f"Context:\n{'\n\n'.join(ctx_lines)}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )    
    return prompt


def generate_answer(query: str, retrieved: List[Dict]) -> str:
    """
    Main generation entrypoint used by FastAPI.
    Builds prompt from retrieved chunks, calls Gemini, then cleans up answer.
    """
    prompt = build_prompt(query, retrieved)
    print(f"[RAG] Gemini prompt built, len={len(prompt)} chars")

    text = call_gemini(prompt)

    if not text:
        # If Gemini fails (API error etc.), use local cheap fallback summary
        text = local_fallback_generate(prompt)

    # Heuristic: extract answer after "Answer:" if present
    if "Answer:" in text:
        ans = text.split("Answer:", 1)[-1].strip()
    else:
        ans = text.strip()
    return ans


def call_gemini(prompt: str, max_output_tokens: int = 600, temperature: float = 0.2) -> str | None:
    """
    Call Gemini with a single prompt. Returns plain text or None on failure.
    """
    try:
        
        # print("Gemini key:", GEMINI_API_KEY)
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None
    


def call_hf_inference(model: str, prompt: str, max_new_tokens: int = 400, temperature: float = 0.2):
    print(f"api: {HF_API_TOKEN}")
    if not HF_API_TOKEN or not model:
        print("api token not found")
        return None
    # url = f"https://api-inference.huggingface.co/models/{model}"
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "repetition_penalty": 1.05,
            "return_full_text": False,     # <- avoids echoing the prompt
        },
        "options": {
            "wait_for_model": True,       # <- cold-start models
            "use_cache": True
        }
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()
        # Typical formats from HF Inference:
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return out[0]["generated_text"]
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"]
        # Fallback best-effort
        return json.dumps(out)
    except Exception as e:
        print(f'exception: {e}')
        return None


# ---------- Upsert to FAISS ----------
def upsert_embeddings(rows: List[Dict]):
    # rows = [{"doc_id":..,"doc_title":..,"page":..,"text":..}]
    texts = [r["text"] for r in rows]
    if not texts:
        return []
    vecs = embed_texts(texts)
    index, meta = load_index()

    # Ensure IP index has correct dim
    d = vecs.shape[1]
    if index.d != d:
        # rebuild new index if dim mismatch
        index = faiss.IndexFlatIP(d)
        meta = {"rows": [], "next_row": 0}

    start_row = meta["next_row"]
    index.add(vecs)
    added_ids = list(range(start_row, start_row + len(texts)))
    for rid, r in zip(added_ids, rows):
        newr = {
            "row": rid,
            "doc_id": r["doc_id"],
            "doc_title": r["doc_title"],
            "page": r["page"],
            "text": r["text"],
        }
        meta["rows"].append(newr)
    meta["next_row"] = start_row + len(texts)
    save_index(index, meta)
    return added_ids
