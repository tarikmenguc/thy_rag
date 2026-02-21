"""
=============================================================================
 THY RAG PROJESI — PROJE YOLCULUGU
=============================================================================

Bu dosya, projenin bastan sona nasil insa edildigini anlatan bir
dokumantasyon dosyasidir. Kod calistirmak icin degil, okumak icin.

=============================================================================
 PROJE AMACI
=============================================================================

Turk Hava Yollari Teknik A.S.'nin 2020, 2021, 2022, 2023 yilarina ait
faaliyet raporlari uzerinde soru-cevap yapabilen bir yapay zeka asistani
gelistirmek. Sistem asagidaki ozelliklere sahip olmalidir:

  - Dogal dilde soru sorulabilmeli
  - Cevaplar rapor dokumanlariyla desteklenmeli (kaynak gostermeli)
  - Yil bazinda filtreleme yapilabilmeli
  - Bellek destegi: "Peki ya 2021?" gibi baglama dayali sorular calissin
  - Kendi cevabini degerlendirip gerekirse yeniden arama yapabilsin (Adaptive RAG)

=============================================================================
 ASAMALAR VE IZLENEN YOL
=============================================================================

---------------------------------------------------------------------------
 ASAMA 1 — Temel RAG Mimarisi
---------------------------------------------------------------------------

Gerekenler:
  pip install langchain langchain-groq langchain-huggingface langchain-chroma
  pip install pypdf sentence-transformers chromadb python-dotenv

Adimlar:
  1. PDF dosyalarini oku     → PyPDFLoader
  2. Parcalara bol          → RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
  3. Vektore donustur       → HuggingFaceEmbeddings (paraphrase-multilingual-MiniLM-L12-v2)
  4. Veritabanina kaydet    → Chroma (disk'e persist et)
  5. LLM'i bagla            → ChatGroq (llama-3.3-70b-versatile)
  6. Retrieval zinciri kur  → BaseRetriever + RetrievalQA

Dosya: main.py (ingestion), rag.py (retrieval zinciri - artik kullanilmiyor)

---------------------------------------------------------------------------
 ASAMA 2 — Metadata ve Yil Filtreleme
---------------------------------------------------------------------------

Problem: Her PDF parcasina hangi yila ait oldugu bilinmiyor.
Cozum: PyPDFLoader ile yuklenen her Document'e {"year": "2020"} metadata eklendi.

  doc.metadata["year"] = year  # main.py'de yapilir

Chromadb'ye filtre ile sorgu:
  vectordb.similarity_search(query, filter={"year": {"$in": ["2021", "2022"]}})

---------------------------------------------------------------------------
 ASAMA 3 — Taranan (Scanned) PDF'lerin Okunmasi
---------------------------------------------------------------------------

Problem: 2022 ve 2023 PDF'leri gorsel (scan) formatindaydi.
PyPDFLoader bu dosyalardan hic metin okuyamadi.

Denenen Cozumler:
  - Gemini Vision API  → Gunluk kota asimi (429 TooManyRequests)
  - Groq Vision (llama-4-scout-17b-16e-instruct) → BASARILI

Yaklasim (ingest_scanned_pdfs.py):
  1. Her PDF sayfasi PNG'ye donustur    → fitz (PyMuPDF), DPI=150
  2. PNG'yi base64'e cevir
  3. Groq Vision'a gonder               → HumanMessage(image_url + text)
  4. Cikan metni Document nesnesi yap   → metadata: year, page, extraction_method
  5. ChromaDB'ye ekle

Sonuc:
  2022: 54 sayfa → 49 sayfa okundu → 194 chunk
  2023: 48 sayfa → 45 sayfa okundu → 238 chunk
  Toplam DB: 808 → 1078 chunk

---------------------------------------------------------------------------
 ASAMA 4 — Conversational Memory (Hafiza)
---------------------------------------------------------------------------

Problem: Kullanici "Peki ya cirosu?" dediginde sistem neyi kastettigini anlayamiyordu.
Cozum: Sohbet gecmisi (chat_history) get_answer fonksiyonuna aktarildi.

app.py'de session_state.messages listesinden (user, assistant) ciftleri uretildi
ve rag fonksiyonuna gonderildi.

---------------------------------------------------------------------------
 ASAMA 5 — Re-Ranking (Yeniden Siralama)
---------------------------------------------------------------------------

Problem: Vektör aramasinda cikan 10 dokuman her zaman en alakali degildi.
Cozum: FlashRank kullanilarak 30 dokuman getirilip en alakali 10 secildi.

  pip install flashrank

  from langchain.retrievers import ContextualCompressionRetriever
  from langchain.retrievers.document_compressors import FlashrankRerank

  compressor = FlashrankRerank(top_n=10)
  compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor,
      base_retriever=base_retriever
  )

NOT: Bu ozellik rag.py'deydi, LangGraph versiyonunda temel retriever kullanilmaktadir.

---------------------------------------------------------------------------
 ASAMA 6 — LangGraph ile Adaptive RAG
---------------------------------------------------------------------------

Problem: Sistem bazen yanlis veya uydurma cevaplar veriyordu.
Cozum: Adaptive RAG mimarisi — sistem kendi cevabini degerlendirir.

LangGraph kavrami:
  - State : Tum dugumlerin paylastigi ortak veri sozlugu (TypedDict)
  - Node  : Is yapan Python fonksiyonu (state alir, state guncellemesi doner)
  - Edge  : Dugumler arasi baglanti (dugz veya kosullu)

Grafin Akisi:
  START → [retrieve] → [generate] → [grade_answer] → karar
                          ↑               |
                          └── not_useful ──┘  (max 2 kez)
                                          |
                                          └── useful → END

4 Dugum:
  retrieve(state)     : Vektordb'den dokuman ceker (yil filtresi uygular)
  generate(state)     : LLM ile cevap uretir
  grade_answer(state) : "useful" mi "not_useful" mi karar verir
  decide_next(state)  : "end" veya "retrieve" string'i doner (kosullu kenar)

Dosya: rag_graph.py

=============================================================================
 PROJE DOSYA YAPISI (FINAL)
=============================================================================

  proje_klasoru/
  |
  |-- main.py                  # 2020-2021 PDF'lerini ChromaDB'ye yukler
  |-- ingest_scanned_pdfs.py   # 2022-2023 gorsel PDF'lerini Groq Vision ile yukler
  |-- rag_graph.py             # LangGraph Adaptive RAG motoru (get_answer fonksiyonu)
  |-- app.py                   # Streamlit arayuzu
  |-- .env                     # API anahtarlari (GROQ_API_KEY, GEMINI_API_KEY)
  |-- .gitignore
  |-- README.md
  |-- data/                    # PDF dosyalari (2020.pdf, 2021.pdf, 2022.pdf, 2023.pdf)
  |-- chroma_db_thy/           # Vektör veritabani (1078 chunk)

=============================================================================
 KULLANILAN TEKNOLOJILER VE ROLLER
=============================================================================

  Teknoloji                          | Rol
  -----------------------------------|------------------------------------------
  LangChain                          | RAG zinciri, prompt template, retriever
  LangGraph                          | Adaptive RAG akis kontrolu (graflar)
  Groq (llama-3.3-70b-versatile)     | Ana dil modeli (cevap uretimi, grading)
  Groq (llama-4-scout-17b)           | Vision model (gorsel PDF'den metin cikarma)
  HuggingFace (MiniLM-L12-v2)        | Embedding (metni vektore cevir)
  ChromaDB                           | Vektör veritabani (similarity search)
  Streamlit                          | Web arayuzu
  PyMuPDF (fitz)                     | PDF sayfalarini PNG'ye donusturme
  FlashRank                          | Re-ranking (en alakali dok. secimi)

=============================================================================
 CALISTIRMA SIRASI (SIFIRDAN KURULUM)
=============================================================================

  1. pip install langchain langchain-groq langchain-huggingface langchain-chroma
     pip install pypdf sentence-transformers chromadb python-dotenv
     pip install streamlit pymupdf pillow flashrank langgraph

  2. .env dosyasi olustur:
       GROQ_API_KEY=gsk_...

  3. 2020-2021 PDF'lerini yukle:
       python main.py

  4. 2022-2023 gorsel PDF'lerini yukle (Groq Vision kullanir):
       python ingest_scanned_pdfs.py

  5. Asistani calistir:
       streamlit run app.py

=============================================================================
 OGRENILENLER / ONEMLI NOKTALAR
=============================================================================

  - Dosya adi ile kutuphane adi ayni olmamali! ("langgraph.py" → cakisma!)
  - Taranan PDF'ler icin PyPDFLoader calismiyor, Vision API gerekiyor.
  - Groq free tier: 100K token/gun. Her soru 2 LLM cagrisi yapar (generate + grade).
  - ChromaDB'de filtre: {"year": {"$in": ["2022"]}} formatinda olmali.
  - LangGraph'ta grade_answer'a tam dokuman metni gondermek cok token tuketiyor.
    Sadece soru + cevap gondermek yeterli ve daha verimli.
  - RAGState'teki tum alanlari app.invoke() cagirisinda vermek zorunlu.
"""
