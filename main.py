import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

sys.stdout.reconfigure(encoding='utf-8')

PDF_PATHS = {
    "2020": "data/2020.pdf",
    "2021": "data/2021.pdf",
}
PERSIST_DIR = "./chroma_db_thy"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_pdfs():
    all_docs = []
    for year, path in PDF_PATHS.items():
        if not os.path.exists(path):
            print(f"Dosya bulunamadi: {path}")
            continue
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["year"] = year
        all_docs.extend(docs)
        print(f"{year}: {len(docs)} sayfa yuklendi")
    return all_docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)

def main():
    print("PDF'ler yukleniyor...")
    docs = load_pdfs()
    if not docs:
        print("Hic dokuman yuklenemedi.")
        return

    print("Parcalaniyor...")
    splits = split_docs(docs)
    print(f"{len(docs)} sayfa -> {len(splits)} chunk")

    print("Vektör veritabani olusturuluyor...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(splits, embeddings, persist_directory=PERSIST_DIR)
    print(f"Tamamlandi! {len(splits)} chunk '{PERSIST_DIR}' dizinine kaydedildi.")

if __name__ == "__main__":
    main()