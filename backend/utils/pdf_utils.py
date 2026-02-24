import pdfplumber, re
from PyPDF2 import PdfReader

# def extract_pdf_text_per_page(pdf_path: str):
#     pages = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             try:
#                 text = page.extract_text()
#             except Exception:
#                 text = ""
#             # text = clean_text(text)
#             pages.append((i+1, text))
#     return pages

def extract_pdf_text_per_page(pdf_path: str):
    pages = []
    pdf = PdfReader(pdf_path)
    for i, page in enumerate(pdf.pages):
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        # text = clean_text(text)
        pages.append((i+1, text))
    print("----")
    print("\n".join(p[1] for p in pages[:5]))
    print("----")
    return pages


def clean_text(s: str) -> str:
    s = s.replace('\x00', ' ')  # null bytes
    s = re.sub(r'\s+', ' ', s).strip()
    return s
