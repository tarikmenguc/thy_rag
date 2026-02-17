import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
persist_directory = "./chroma_db_thy"

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

print("DB içindeki döküman sayisi:", vectordb._collection.count())

# Get all unique sources
# This might be slow if DB is huge, but for our size it is fine.
# Chroma doesn't have a direct distinct metadata query easily accessible via LangChain wrapper.
# We will peek at a random sample.
# Fetch all metadata to count sources manually
print("\nTüm döküman kaynakları sayılıyor...")
all_data = vectordb.get(include=["metadatas"])
source_counts = {}

for meta in all_data['metadatas']:
    source = meta.get('source', 'Unknown')
    filename = os.path.basename(source)
    source_counts[filename] = source_counts.get(filename, 0) + 1

print("\nKaynak Dağılımı:")
for source, count in source_counts.items():
    print(f"- {source}: {count} chunk")

if "2022.pdf" in source_counts:
    print(f"\n✅ 2022.pdf veritabanında mevcut ({source_counts['2022.pdf']} chunk).")
else:
    print("\n❌ UYARI: 2022.pdf veritabanında BULUNAMADI!")
