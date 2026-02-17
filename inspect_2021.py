import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
persist_directory = "./chroma_db_thy"

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

print("2022.pdf içeriği getiriliyor...")
# Chroma doesn't support $like, so we fetch all metadatas and filter in python, then fetch by ID
# This is inefficient but fine for debugging 800 docs.

all_data = vectordb.get(include=["metadatas", "documents"])

count = 0
for i, meta in enumerate(all_data['metadatas']):
    source = meta.get('source', '')
    if "2021.pdf" in source:
        print(f"\n--- Chunk {count+1} (Source: {source} Page: {meta.get('page')}) ---")
        print(all_data['documents'][i][:500]) # Print first 500 chars
        count += 1
        if count >= 10: # Just print first 10 chunks to check quality
            break
