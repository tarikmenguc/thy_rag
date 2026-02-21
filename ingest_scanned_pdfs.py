"""
ingest_scanned_pdfs.py

Taranan (gorsel tabanli) PDF dosyalarini (2022 ve 2023 raporlari)
Groq Llama Vision API kullanarak metne cevirir ve mevcut ChromaDB 
vektör veritabanina ekler.
"""

import os
import io
import sys
import time
import base64
import fitz  # PyMuPDF
from dotenv import load_dotenv
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# --- Yapilandirma ---
SCANNED_PDFS = {
    "2022": "C:/Users/tarik/Desktop/proje_klasoru/data/2022.pdf",
    "2023": "C:/Users/tarik/Desktop/proje_klasoru/data/2023.pdf",
}
PERSIST_DIR = "./chroma_db_thy"

# --- Groq Vision Model ---
groq_api_key = os.getenv("GROQ_API_KEY")
vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=groq_api_key,
    temperature=0
)
print("Groq Vision LLM baglandi: llama-4-scout-17b-16e-instruct")

# --- Embedding ve ChromaDB ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
print(f"Mevcut veritabani yuklendi: {PERSIST_DIR}")

# --- Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)


# --- Yardimci Fonksiyonlar ---
def pdf_page_to_base64(pdf_path: str, page_num: int) -> str:
    """PDF sayfasini base64 PNG formatina donusturur."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)  # 150 DPI: token/kalite dengesi
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_text_from_page(pdf_path: str, page_num: int, year: str) -> str:
    """Groq Vision LLM ile PDF sayfasindan metin cikarir."""
    b64_image = pdf_page_to_base64(pdf_path, page_num)

    prompt = f"""{year} yili Turk Hava Yollari Teknik A.S. faaliyet raporunun {page_num + 1}. sayfasidir.

Sayfadaki TUM metni, rakamlari ve tablo iceriklerini eksiksiz cikar.
- Tablolardaki her hucreyı ayri satira yaz.
- Sayisal degerleri oldugu gibi koru.
- Sadece sayfadaki metni ver, yorum ekleme.
- Sayfa bos veya sadece gorsel iceriyorsa 'SAYFA_BOS' yaz."""

    max_retries = 4
    for attempt in range(max_retries):
        try:
            message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                },
                {"type": "text", "text": prompt}
            ])
            response = vision_llm.invoke([message])
            return response.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_time = 30 * (attempt + 1)
                print(f"\n    [Rate limit] {wait_time}s bekleniyor...", end=" ", flush=True)
                time.sleep(wait_time)
            else:
                print(f"\n    Hata (Sayfa {page_num}): {e}")
                return ""
    return ""


def process_scanned_pdf(pdf_path: str, year: str) -> list:
    """Taranan PDF'i sayfa sayfa isler, Document listesi doner."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    all_docs = []
    print(f"\nIsleniyor: {os.path.basename(pdf_path)} ({total_pages} sayfa)")

    for page_num in range(total_pages):
        print(f"  Sayfa {page_num + 1}/{total_pages}...", end=" ", flush=True)

        text = extract_text_from_page(pdf_path, page_num, year)

        if not text.strip() or text.strip() == "SAYFA_BOS":
            print("bos, atlandi.")
            continue

        doc_obj = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_num,
                "year": year,
                "extraction_method": "groq_vision"
            }
        )
        all_docs.append(doc_obj)
        print(f"OK ({len(text)} karakter)")

        # Rate limiting: Groq free tier = 30 req/min
        time.sleep(3)

    return all_docs


# --- Ana Islem ---
if __name__ == "__main__":
    all_new_docs = []

    for year, pdf_path in SCANNED_PDFS.items():
        if not os.path.exists(pdf_path):
            print(f"HATA: Dosya bulunamadi: {pdf_path}")
            continue

        docs = process_scanned_pdf(pdf_path, year)
        all_new_docs.extend(docs)
        print(f"\n  [{year}] {len(docs)} sayfa basariyla islendi.")

    if not all_new_docs:
        print("\nHic dokuman uretilemedi. PDF yollarini kontrol edin.")
        sys.exit(1)

    print(f"\nMetinler parcalaniyor (chunking)...")
    splits = text_splitter.split_documents(all_new_docs)
    print(f"  {len(all_new_docs)} sayfa -> {len(splits)} chunk")

    print(f"\nChromaDB'ye ekleniyor...")
    vectordb.add_documents(splits)
    print(f"  TAMAM: {len(splits)} chunk veritabanina eklendi!")

    total_count = vectordb._collection.count()
    print(f"\nTamamlandi! Veritabanindaki toplam chunk sayisi: {total_count}")
    print("Artik 2022 ve 2023 raporlarindan da soru sorabilirsiniz.")
