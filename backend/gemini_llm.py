import google.generativeai as genai
from . import settings

if not settings.GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

genai.configure(api_key=settings.GEMINI_API_KEY)

def generate_answer_with_gemini(question: str, context: str) -> str:
    """
    Simple helper that uses Gemini to answer a question using given context.
    """
    prompt = f"""
You are an AI assistant answering questions strictly using the given context.
If the answer is not clearly in the context, say you do not know.

Question:
{question}

Context (from research paper):
{context}
"""

    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    resp = model.generate_content(prompt)
    # resp.text usually contains the answer
    return resp.text.strip() if hasattr(resp, "text") else str(resp)
