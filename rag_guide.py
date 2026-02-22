"""
=============================================================================
 RAG SİSTEMLERİ TASARIM REHBERİ
=============================================================================

Her RAG projesinde sorulması gereken sorular, uygulanması gereken teknikler,
ve hangi durumda hangi yöntemin seçileceğine dair pratik bir referans kılavuz.

Okumak için açın, çalıştırmak için değil.
=============================================================================
"""

# =============================================================================
# BÖLÜM 1: TEMEL RAG MİMARİSİ (Her projede zorunlu)
# =============================================================================

TEMEL_MIMARI = """
Her RAG sisteminin 3 zorunlu katmanı vardır:

1. INGESTION (Veri Hazırlama)
   Belgeler → Chunk → Embedding → Vector DB

2. RETRIEVAL (Arama)
   Kullanıcı Sorusu → Embedding → Benzerlik Araması → İlgili Chunk'lar

3. GENERATION (Cevap Üretme)
   Soru + Chunk'lar → LLM Prompt → Cevap


Minimum çalışan bir RAG için gereken kütüphaneler:

   from langchain_community.document_loaders import PyPDFLoader
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain_huggingface import HuggingFaceEmbeddings  # veya OpenAI
   from langchain_chroma import Chroma                       # veya FAISS, Pinecone
   from langchain_groq import ChatGroq                       # veya OpenAI, Ollama


Sıfırdan kurulum (her projede tekrarlanır):

   1. Belgeleri yükle
   2. Chunk'lara böl (chunk_size, chunk_overlap belirle)
   3. Embedding modeliyle vektörize et
   4. ChromaDB'ye kaydet (persist_directory belirt!)
   5. Retriever oluştur
   6. LLM + Prompt + Retriever → Chain
"""


# =============================================================================
# BÖLÜM 2: CHUNK STRATEJİSİ (En kritik karar)
# =============================================================================

CHUNK_STRATEJISI = """
Chunk boyutu RAG'ın en kritik hiperparametresidir.
Yanlış boyut → yanlış retrieval → yanlış cevap.

KURAL: Küçük chunk = hassas arama, büyük chunk = zengin bağlam

─────────────────────────────────────────────────────
 chunk_size  │  overlap   │  Ne zaman kullanılır?
─────────────────────────────────────────────────────
   200-400   │   50-100   │ Teknik dokümanlar, tablolar,
             │            │ kısa cümle yoğun içerik
─────────────────────────────────────────────────────
   500-1000  │  100-200   │ Genel dokümanlar, pdf raporlar,
             │            │ makaleler (en yaygın seçim)
─────────────────────────────────────────────────────
  1500-2500  │  200-400   │ Hukuk belgeleri, sözleşmeler,
             │            │ uzun anlatı metinler
─────────────────────────────────────────────────────
  PARENT-CHILD:
   Child: 300  │ 50   │ Arama için (hassas eşleşme)
  Parent: 2000 │ 200  │ LLM bağlamı için (zengin içerik)
─────────────────────────────────────────────────────

⚠️  Pratikte chunk boyutunu değiştirmek = veri tabanını yeniden oluşturmak demek.
    İlk başta düşünerek karar ver, sonradan değiştirmek maliyetli.


SEPARATOR SEÇİMİ:

   separators = ["\n\n", "\n", ". ", " "]
   #              ↑         ↑       ↑
   #           paragraf  satır  cümle  (öncelik sırası)

   Kod içeren belgeler için:
   separators = ["\n\n", "\n", "}", "{", " "]
"""


# =============================================================================
# BÖLÜM 3: EMBEDDING MODELİ SEÇİMİ
# =============================================================================

EMBEDDING_MODELI = """
Embedding modeli = Arama kalitesi = RAG kalitesi

ÜCRETSİZ (local çalışır):
   paraphrase-multilingual-MiniLM-L12-v2   → Türkçe dahil çok dilli (önerilir)
   all-MiniLM-L6-v2                        → İngilizce, hızlı, hafif
   BAAI/bge-m3                             → Çok dilli, güçlü, yavaşça

ÜCRETLİ (API):
   text-embedding-3-small (OpenAI)         → İngilizce'de en iyi
   text-embedding-3-large (OpenAI)         → Pahalı ama çok iyi

KURAL: Türkçe içerik → mutlaka multilingual model kullan.
       Paraphrase modeller, farklı ifade edilen aynı anlamlı metinleri bulur.
       Similarity modeller, birebir kelime benzerliğine bakar.

⚠️  Embedding modeli değiştirirsen → ChromaDB'yi sıfırdan oluşturman gerekir.
"""


# =============================================================================
# BÖLÜM 4: VERİTABANI KARARLARI
# =============================================================================

VERITABANI_KARARLARI = """
YEREL (geliştirme / demonstrasyon):
   ChromaDB     → Kurulum gerektirmez, persist_directory ile kalıcı
   FAISS        → RAM'de çalışır, hızlı ama dinamik ekleme zor

BULUT (production):
   Pinecone     → Yönetilen, ölçeklenebilir, ücretli
   Weaviate     → Açık kaynak, kendi sunucunda kurulabilir
   Qdrant       → Yüksek performans, Rust tabanlı

KARAR TABLOSU:
   ┌────────────────────┬────────────┬────────────┬───────────┐
   │ İhtiyaç            │ ChromaDB   │ FAISS      │ Pinecone  │
   ├────────────────────┼────────────┼────────────┼───────────┤
   │ Proje / POC        │ ✅ İdeal   │ ✅ İdeal   │ ❌ Fazla  │
   │ Küçük veri (<100K) │ ✅         │ ✅         │ ✅        │
   │ Büyük veri (>1M)   │ ⚠️ Yavaş  │ ⚠️ RAM    │ ✅        │
   │ Metadata filtresi  │ ✅         │ ❌         │ ✅        │
   │ Gerçek zamanlı ek  │ ✅         │ ❌ Zor     │ ✅        │
   └────────────────────┴────────────┴────────────┴───────────┘

NEDEN AYRI DB KULLANDIK? (Parent-Child için)
   Mevcut chroma_db_thy içinde 1000-karakter chunk'lar var.
   Parent-child şeması 300/2000 karakter istiyor.
   Aynı koleksiyona farklı boyutlarda chunk koyarsan:
   → Embedding uzayı karışır
   → Retrieval kalitesi düşer
   → Hangi chunk nasıl üretildi bilemezsin
   Bu yüzden: ayrı collection, ayrı persist_directory.
"""


# =============================================================================
# BÖLÜM 5: GELİŞMİŞ RAG TEKNİKLERİ
# =============================================================================

GELISMIS_TEKNIKLER = """
Temel RAG'ı kurduktan sonra sırayla eklenebilecek teknikler:

─────────────────────────────────────────────────────────────────
1. METADATA FİLTRELEME
─────────────────────────────────────────────────────────────────
   Nedir?  : doc.metadata["year"] = "2022" gibi etiketler ekle,
             aramalarda filtrele.
   Ne zaman: Çok yıllık / çok kaynaklı veri varsa.
   Nasıl   :
             vectordb.similarity_search(query, filter={"year": "2022"})
   Uyarı   : Filtre çok kısıtlayıcı olursa hiç sonuç gelmez.

─────────────────────────────────────────────────────────────────
2. RE-RANKING (Yeniden Sıralama)
─────────────────────────────────────────────────────────────────
   Nedir?  : 10 chunk getir, en alakalı 3'ü seç.
   Ne zaman: Retrieval skoru yeterli değilse, false positive çoksa.
   Nasıl   : FlashRank, Cohere Rerank, Cross-encoder modeller
   Uyarı   : Ekstra gecikme (~100-300ms) ekler.

─────────────────────────────────────────────────────────────────
3. HyDE (Hypothetical Document Embedding)
─────────────────────────────────────────────────────────────────
   Nedir?  : LLM'e sorudan hayali bir cevap ürettir, o cevabı ara.
   Ne zaman: Kısa sorular ile uzun dokümanlar arasındaki
             semantik uçurum varsa.
   Ne zaman kullanma: Spesifik sayısal sorgularda ters tepebilir.
   Nasıl   :
             hypothesis = llm("Bu soruya kisa cevap yaz: {question}")
             docs = vectordb.similarity_search(hypothesis)

─────────────────────────────────────────────────────────────────
4. PARENT-CHILD CHUNKING
─────────────────────────────────────────────────────────────────
   Nedir?  : Küçük chunk'la ara, büyük chunk'ı LLM'e gönder.
   Ne zaman: Cevaplar eksik/bağlamsız geliyorsa.
             Dokümanlar bilgiyi birden fazla cümlede anlatıyorsa.
             Uzun teknik raporlar, hukuk belgeleri.
   Ne zaman kullanma: Kısa belgeler veya tablo ağırlıklı veriler,
                      parent chunk fazla gürültü içerir.

─────────────────────────────────────────────────────────────────
5. ADAPTIVE RAG / LangGraph
─────────────────────────────────────────────────────────────────
   Nedir?  : Cevap iyi değilse tekrar ara (self-correction loop).
   Akış    : retrieve → generate → grade → (not_useful?) → retrieve
   Ne zaman: Retrieval tutarsızsa, bazı sorular sürekli kötü cevap
             alıyorsa.
   Uyarı   : Her döngü = ekstra LLM çağrısı = ekstra token/ücret.
             max_retries = 2 gibi bir sınır koy.

─────────────────────────────────────────────────────────────────
6. MULTI-QUERY
─────────────────────────────────────────────────────────────────
   Nedir?  : Aynı soruyu farklı şekillerde ifade et, hepsini ara,
             sonuçları birleştir.
   Ne zaman: Tek soru zayıf retrieval veriyorsa.
   Nasıl   :
             Sorudan 3 farklı versiyon üret → 3 ayrı arama yap →
             toplam sonuçları birleştir → LLM'e gönder.

─────────────────────────────────────────────────────────────────
7. CONVERSATIONAL MEMORY
─────────────────────────────────────────────────────────────────
   Nedir?  : Önceki soruları bağlama dahil et.
   Ne zaman: Chatbot tarzı uygulamalarda her zaman.
   Nasıl   :
             chat_history = [(soru1, cevap1), (soru2, cevap2)]
             prompt → chat_history + yeni soru

─────────────────────────────────────────────────────────────────
8. GÖRSEL PDF / OCR
─────────────────────────────────────────────────────────────────
   Nedir?  : Taranmış PDF'leri Vision API ile metne çevir.
   Ne zaman: PDF metin içermiyor, PyPDF boş string dönüyorsa.
   Nasıl   : PyMuPDF ile sayfa → PNG → Groq/Claude Vision → metin
   Uyarı   : Pahalı (token başına) ve yavaş.
"""


# =============================================================================
# BÖLÜM 6: DEĞERLENDİRME (Evaluation)
# =============================================================================

DEGERLENDIRME = """
RAG sistemini ölçmeden iyileştirdiğini bilemezsin.

TEMEL METRİKLER:
   Correctness  : Cevap doğru bilgiyi içeriyor mu? (0/1)
   Faithfulness : Cevap kaynaklara sadık mı? (uydurmuyor mu?)
   Relevance    : Getirilen chunk'lar soruyla alakalı mı?
   Completeness : Beklenen tüm bilgi verildi mi?

YÖNTEMLİR:
   1. LLM-as-Judge (bu projede kullandık):
         LLM'e "bu cevap doğru mu?" sorusu sorulur.
         Hızlı ve esnek ama LLM'nin hatalarını miras alır.

   2. RAGAS Framework:
         pip install ragas
         Otomatik olarak Faithfulness, Answer Relevancy,
         Context Recall hesaplar.

   3. İnsan Değerlendirmesi (Annotation Queues):
         LangSmith → Annotation Queues
         Her cevabı thumbs up/down ile işaretle.
         En güvenilir ama en yavaş yöntem.

LANGSMITH İKİLİ YAPISI:
   Tracing          → Her çalışmada otomatik. Monitoring için.
   Datasets & Exps  → Bilinçli test. Karşılaştırma için.
   İkisi ayrı amaçlara hizmet eder, birbirini tamamlar.

NE ZAMAN EVALUATE ÇALIŞTIR?
   → Yeni bir teknik ekledikten sonra (HyDE, Parent-Child)
   → Prompt değiştirdikten sonra
   → Embedding modeli değiştirdikten sonra
   → Chunk boyutu değiştirdikten sonra
   → Production'a geçmeden önce
"""


# =============================================================================
# BÖLÜM 7: PROJE BAŞLARKEN SORULACAK SORULAR
# =============================================================================

BASLANGIC_SORULARI = """
Yeni bir RAG projesi başlamadan önce şu soruların cevabı açık olmalı:

1. VERİ YAPISI
   □ Belgeler metin mi yoksa taranmış görsel mi?
   □ Kaç belge, toplam kaç sayfa?
   □ Dil nedir? (Türkçe → multilingual embedding zorunlu)
   □ Tablolar, grafikler var mı? → Özel işlem gerekebilir.

2. SORU TİPLERİ
   □ Sorular kısa mu (1-5 kelime) yoksa uzun mu?
     → Kısaysa HyDE düşün.
   □ Spesifik veri mi soruluyor (sayı, tarih) yoksa genel bilgi mi?
     → Spesifikse metadata filtreleme zorunlu.
   □ Sohbet tarzı mı yoksa tek seferlik mi?
     → Sohbetse conversational memory ekle.

3. PERFORMANS GEREKSİNİMLERİ
   □ Gecikme sınırı nedir? (<2s, <5s, <10s?)
     → Çok düşükse: küçük embedding modeli, az chunk.
   □ Kaç eş zamanlı kullanıcı?
     → Çok fazlaysa: bulut vektör DB gerekli.

4. BÜTÇE
   □ API bedeli var mı?
     → Yoksa: yerel LLM (Ollama), yerel embedding.
   □ Token sınırı var mı?
     → Grade_answer sorgusunda bağlamı kısalt.

5. DOĞRULUK GEREKSİNİMİ
   □ Yanlış cevap kabul edilebilir mi?
     → Kritikse: RAG + kaynak gösterme zorunlu.
   □ Kaynak doğrulama gerekli mi?
     → Faithfulness evaluator ekle.
"""


# =============================================================================
# BÖLÜM 8: BU PROJEDEKİ MİMARI (Örnek)
# =============================================================================

BU_PROJENIN_MIMARISI = """
THY Teknik RAG Projesi — Seçilen Mimari ve Gerekçeleri:

SORUN              ÇÖZÜM              GEREKÇE
────────────────────────────────────────────────────────────────────
Türkçe içerik    │ Multilingual      │ Türkçe embedding modeli olm
                 │ MiniLM-L12-v2    │ İngilizce modeller Türkçe'de zayıf
─────────────────┼───────────────────┼────────────────────────────
Taranmış PDF     │ Groq Vision OCR  │ PyPDF metin çıkaramıyor
(2022-2023)      │ llama-4-scout    │ Ücretsiz, hızlı
─────────────────┼───────────────────┼────────────────────────────
Yıla göre arama  │ Metadata filter  │ "2022 projesi" sorması için
                 │ year: "2022"     │ tüm yılları karıştırmamak
─────────────────┼───────────────────┼────────────────────────────
Kısa soru →      │ HyDE             │ "net kar" → kısa query,
uzun doküman    │                   │ dokümanlarda "85.455.249 TL"
                 │                   │ gibi uzun bağlam var
─────────────────┼───────────────────┼────────────────────────────
Cevap bağlamı   │ Parent-Child     │ 1000-char chunk yetmez,
az geliyor       │ Child=300        │ 2000-char parent daha zengin
                 │ Parent=2000      │ bağlam sağlar
─────────────────┼───────────────────┼────────────────────────────
Kötü cevap       │ Adaptive RAG     │ Grade=not_useful → tekrar ara
                 │ LangGraph        │ Max 2 retry, crash önler
─────────────────┼───────────────────┼────────────────────────────
Token limiti     │ grade_answer'da  │ Groq free tier 100K/gün
                 │ kısa prompt      │ Her soru 2-3 LLM çağrısı
─────────────────┼───────────────────┼────────────────────────────
Monitoring       │ LangSmith        │ Her trace otomatik kayıt
                 │ LANGSMITH_       │ Hangi node kaç ms, kaç token
                 │ TRACING=true     │
─────────────────┼───────────────────┼────────────────────────────
Değerlendirme    │ LangSmith        │ 5 soruluk dataset, LLM-judge
                 │ Datasets &       │ HyDE öncesi/sonrası karşılaş
                 │ Experiments      │ tirması yapılabilir
────────────────────────────────────────────────────────────────────


Seçilmeyen teknikler ve nedenleri:

RAPTOR:
   Hiyerarşik özetleme → anlamlı ama karmaşık.
   57 sayfalık rapor için ezber kaydetme değil.
   → Atlandı.

FAISS:
   Metadata filtreleme yok, dinamik ekleme zor.
   → ChromaDB tercih edildi.

OpenAI Embedding:
   İngilizce dominant, Türkçe için suboptimal + ücretli.
   → Multilingual HuggingFace seçildi.
"""

if __name__ == "__main__":
    print("Bu dosya dokümantasyon amacıyla yazılmıştır.")
    print("Okumak için doğrudan açın veya bir Python okuyucu kullanın.")
