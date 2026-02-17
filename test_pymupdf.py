from langchain_community.document_loaders import PyMuPDFLoader
import os

pdf_path = "data/2022.pdf"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

print(f"Total pages extracted: {len(docs)}")
total_chars = sum([len(doc.page_content) for doc in docs])
print(f"Total characters extracted: {total_chars}")

print("\n--- First 500 chars of Page 1 ---")
if len(docs) > 0:
    print(docs[0].page_content[:500])

print("\n--- First 500 chars of Last Page ---")
if len(docs) > 0:
    print(docs[-1].page_content[:500])
