"""
Parent-Child Chunking Ingestion (Manuel Implementasyon)
---------------------------------------------------------
Bu script, 2020 ve 2021 PDF'lerini Parent-Child yontemi ile ChromaDB'ye yukler.

Yaklasim:
  - Her sayfa 2000-karakterlik parent chunk'lara bolunur
  - Her parent 300-karakterlik child chunk'lara bolunur
  - Child chunk'lar ChromaDB'ye kaydedilir
  - Her child'in metadata'sinda parent_content saklanir
  - Arama sirasinda child bulunur, LLM'e parent gonderilir

Neden ParentDocumentRetriever kullanmiyoruz?
  langchain versiyonumuzda deprecated modul cakismasi var.
  Manuel implementasyon daha egitici ve esnek.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

PDF_PATHS = {
    "2020": "data/2020.pdf",
    "2021": "data/2021.pdf",
}
PC_PERSIST_DIR  = "./chroma_db_thy_pc"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE  = 300

def load_pdfs():
    all_docs = []
    for year, path in PDF_PATHS.items():
        if not os.path.exists(path):
            print(f"  Dosya bulunamadi: {path}")
            continue
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["year"] = year
        all_docs.extend(docs)
        print(f"  {year}: {len(docs)} sayfa yuklendi")
    return all_docs

def main():
    print("Embeddings yukleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    print("PDF'ler yukleniyor...")
    docs = load_pdfs()
    if not docs:
        print("Hic dokuman yuklenemedi.")
        return

    print("Parent-Child chunking yapiliyor...")
    child_docs = []
    parent_count = 0
    child_count  = 0

    parents = parent_splitter.split_documents(docs)
    parent_count = len(parents)

    for parent in parents:
        parent_id = str(uuid.uuid4())
        children  = child_splitter.split_text(parent.page_content)

        for child_text in children:
            child_doc = Document(
                page_content=child_text,
                metadata={
                    **parent.metadata,
                    "parent_id":      parent_id,
                    "parent_content": parent.page_content,  # LLM'e bu gonderilecek
                    "chunk_type":     "child"
                }
            )
            child_docs.append(child_doc)
            child_count += 1

    print(f"  {parent_count} parent → {child_count} child chunk olusturuldu")
    print("ChromaDB'ye kaydediliyor...")

    Chroma.from_documents(
        child_docs,
        embeddings,
        persist_directory=PC_PERSIST_DIR,
        collection_name="thy_parent_child"
    )
    print(f"Tamamlandi! '{PC_PERSIST_DIR}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()
