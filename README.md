# Turkish Technic RAG Asistanı ✈️

**Türk Hava Yolları Teknik A.Ş.**'nin 2020–2023 yıllarına ait faaliyet raporları üzerinde soru-cevap yapabilen **Adaptive RAG** uygulaması.

## 🎯 Proje Hedefi ve Öğrenilenler

Bu projeyi Generative AI ve RAG sistemlerini öğrenmek amacıyla geliştirdim.

**Kapsanan Konular:**
- **RAG Mimarisi** — dış belgelerden bağlam alarak LLM cevaplarını doğrulama
- **Vektör Veritabanı** — ChromaDB ile metin parçalarının embedding'e dönüştürülmesi ve sorgulanması
- **LangGraph** — düğüm/kenar tabanlı stateful graf ile Adaptive RAG akışı
- **Vision AI** — taranmış (scanned) PDF'lerden Groq Vision ile metin çıkarma
- **Metadata Filtreleme** — ChromaDB'de yıl bazlı döküman filtreleme
- **Conversational Memory** — sohbet geçmişiyle bağlama dayalı soru anlama
- **Re-Ranking** — FlashRank ile en alakalı dökümanların seçimi

## 🏗️ Mimari

```
Kullanıcı Sorusu
      │
      ▼
 [retrieve]  ← ChromaDB'den ilgili dökümanları çeker (yıl filtresiyle)
      │
      ▼
 [generate]  ← Groq LLM ile cevap üretir
      │
      ▼
[grade_answer] ← Cevabı değerlendirir: useful mi, not_useful mi?
      │
      ├── useful     → Kullanıcıya gönder ✅
      └── not_useful → [retrieve]'e geri dön 🔄 (max 2 kez)
```

## 🛠️ Teknoloji Yığını

| Teknoloji | Rol |
|---|---|
| **LangGraph** | Adaptive RAG akış kontrolü (graf + döngü) |
| **LangChain** | Prompt template, retriever, zincir oluşturma |
| **Groq** (`llama-3.3-70b-versatile`) | Ana dil modeli |
| **Groq** (`llama-4-scout-17b`) | Vision — taranmış PDF'den metin çıkarma |
| **ChromaDB** | Vektör veritabanı (1078 chunk) |
| **HuggingFace** (`paraphrase-multilingual-MiniLM-L12-v2`) | Çok dilli embedding |
| **Streamlit** | Sohbet arayüzü |
| **PyMuPDF** | PDF → PNG dönüşümü (OCR için) |
| **FlashRank** | Re-ranking (en alakalı dökümanlar) |

## 🚀 Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/tarikmenguc/thy_rag.git
cd thy_rag

# 2. Bağımlılıkları kur
pip install langchain langchain-groq langchain-huggingface langchain-chroma
pip install chromadb pypdf sentence-transformers python-dotenv
pip install streamlit pymupdf pillow flashrank langgraph

# 3. .env dosyası oluştur
echo GROQ_API_KEY=gsk_... > .env

# 4. 2020-2021 PDF'lerini yükle (metin tabanlı)
python main.py

# 5. 2022-2023 PDF'lerini yükle (taranmış → Groq Vision)
python ingest_scanned_pdfs.py

# 6. Uygulamayı başlat
streamlit run app.py
```

## 📂 Proje Yapısı

```
thy_rag/
├── main.py                  # 2020-2021 PDF ingestion
├── ingest_scanned_pdfs.py   # 2022-2023 görsel PDF ingestion (Groq Vision OCR)
├── rag_graph.py             # LangGraph Adaptive RAG motoru
├── app.py                   # Streamlit sohbet arayüzü
├── project_journey.py       # Proje geliştirme yolculuğu (dokümantasyon)
├── .env                     # API anahtarları
├── data/                    # PDF raporları (2020–2023)
└── chroma_db_thy/           # Vektör veritabanı
```

## 💬 Kullanım

Uygulama açıldığında sol panelden analiz edilecek yılları seçin, ardından sorularınızı yazın:

- *"2022 yılında hangi uçak tipleri için bakım yapıldı?"*
- *"2023'teki çalışan sayısı ve unvan dağılımı nedir?"*
- *"2021 net kârı ne kadardı? Peki ya 2020?"* ← hafıza desteği

## ⚠️ Notlar

- **Free Tier Limiti:** Groq ücretsiz katmanında günlük 100K token sınırı var. Her soru 2 LLM çağrısı yapar (generate + grade).
- **2022-2023 İçerikleri:** Bu raporlar operasyonel niteliktedir; finansal tablo içermez. Teknik faaliyetler, sertifikalar ve müşteri bilgileri mevcuttur.
- **Dosya adı çakışması:** `langgraph.py` adında dosya oluşturulmamalı — Python `langgraph` kütüphanesiyle çakışır.
