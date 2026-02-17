import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
data_dir="C:/Users/tarik/Desktop/proje_klasoru/data"
all_docs = []
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("Dosyalar işleniyor...")
for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
        file_path=os.path.join(data_dir,file)
        loader=PyPDFLoader(file_path)
        pages=loader.load()

        year=file.replace(".pdf","")

        for page in pages:
            page.metadata["year"]=year
            all_docs.append(page)  

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)
splits=text_splitter.split_documents(all_docs)

vectorstore=Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db_thy"
)
print("veritabani kaydedildi")