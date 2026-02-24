import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/sudharshan.db")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
GEMINI_API_KEY = "AIzaSyAx8w87lKy6psIf54RykhCAWyWvMe5H2ms"
# GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

FALLBACK_GENERATION_MODEL = os.getenv("FALLBACK_GENERATION_MODEL", "google/flan-t5-base")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = os.getenv("INDEX_DIR", "./data")
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "3"))
