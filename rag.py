import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

# Logları ayarla (sadece hataları göster)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# .env dosyasını yükle
load_dotenv()

# 1. Groq ve Embedding Ayarları
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Vektör Veritabanını Yükle
persist_directory = "./chroma_db_thy"
if not os.path.exists(persist_directory):
    print("HATA: Vektör veritabanı bulunamadı! Önce main.py dosyasını çalıştırın.")
    exit()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 3. Multi-query Retriever (Akıllı Sorgulama)
# Kullanıcının sorusunu farklı açılardan 3 farklı soruya dönüştürür
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}), llm=llm
)

# 4. Prompt Template (Sistem Mesajı)
template = """Aşağıdaki bağlam bilgisini kullanarak soruyu cevapla.
Cevabı bilmiyorsan "Bilmiyorum" de, uydurma.
Mümkün olduğunca detaylı ve açıklayıcı ol.

Bağlam:
{context}

Soru: {question}

Cevap:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. Retrieval Chain Oluşturma
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_from_llm,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
)

def get_answer(query):
    """
    Sorguyu alır ve cevabı (kaynaklarla birlikte) döner.
    """
    return qa_chain.invoke({"query": query})

def ask_question(query):
    print(f"\n🚀 Soru: {query}")
    print("-" * 50)
    
    # Cevabı al
    result = get_answer(query)
    
    print(f"🤖 Cevap: {result['result']}\n")
    
    print("📚 Kaynaklar:")
    seen_sources = set()
    for doc in result['source_documents']:
        source = doc.metadata.get('source', 'Bilinmiyor')
        page = doc.metadata.get('page', 0)
        # Sadece dosya adını al (path'i temizle)
        filename = os.path.basename(source)
        
        source_key = f"{filename} - Sayfa {page}"
        if source_key not in seen_sources:
            print(f"- {source_key}")
            seen_sources.add(source_key)
    print("-" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        ask_question(query)
    else:
        while True:
            query = input("\nSorunuzu yazın (Çıkış için 'q'): ")
            if query.lower() == 'q':
                break
            ask_question(query)
