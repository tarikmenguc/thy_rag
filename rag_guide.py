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
# BÖLÜM 5B: İLERİ SEVİYE RAG TEKNİKLERİ (Detaylı)
# =============================================================================

ILERI_TEKNIKLER = """
─────────────────────────────────────────────────────────────────
SELF-RAG — Modelin kendi kararını vermesi
─────────────────────────────────────────────────────────────────
Nedir?
   Modelin her adımda "ne yapmalıyım?" kararını bizzat vermesidir.
   Standart RAG'da akış sabittir: her zaman ara, her zaman üret.
   Self-RAG'da model şunu sorar:
      "Bu soruya cevap vermek için kaynak aramam gerekiyor mu?"
      "Getirdiğim chunk'lar alakalı mı?"
      "Ürettiğim cevap kaynaklara sadık mı?"

Nasıl çalışır?
   Model şu özel token'ları üretir:
   [Retrieve]      → arama gerekli mi?
   [IsREL]         → chunk alakalı mı?
   [IsSUP]         → cevap chunk'larla destekleniyor mu?
   [IsUSE]         → cevap kullanıcıya yararlı mı?

   Örnek akış:
   Soru gelir → Model [Retrieve]=YES derse → ara → [IsREL] değerlendir
             → [Retrieve]=NO derse → direk cevap üret (arama yok!)

Bu projede ne yaptık?
   Adaptive RAG ile benzer: grade_answer node'u "useful/not_useful"
   kararı veriyor. Bu Self-RAG'ın basitleştirilmiş hali.
   Fark: gerçek Self-RAG özel fine-tune edilmiş model gerektirir.

Ne zaman kullan?
   → Bazı sorular için kesinlikle kaynak gerekli, bazıları için değil.
     Örnek: "Merhaba" → arama gereksiz | "2022 kârı ne?" → arama gerekli.
   → Token maliyetini düşürmek istiyorsan.

Zorluk: ⭐⭐⭐
   Fine-tuned model (Llama-2-7b-self-rag) veya karmaşık LangGraph
   yapısı gerektirir. Başlangıç seviyesi için önerilmez.

Kod iskelet:
   def should_retrieve(state):
       prompt = "Bu soruya cevap vermek için belge araması gerekiyor mu? YES/NO"
       decision = llm.invoke(prompt + state["question"])
       return "retrieve" if "YES" in decision else "generate_direct"

   workflow.add_conditional_edges(START, should_retrieve, {
       "retrieve": "retrieve",
       "generate_direct": "generate"
   })


─────────────────────────────────────────────────────────────────
RAPTOR — Hiyerarşik Özetleme ile İndeksleme
─────────────────────────────────────────────────────────────────
Nedir?
   Recursive Abstractive Processing for Tree-Organized Retrieval.
   Belgeler hiyerarşik bir ağaç yapısında indekslenir:
   Belge → Küçük chunk'lar (yaprak)
         → Küçük chunk özetleri (ara düğüm)
         → Genel belge özeti (kök)

   Arama yapılırken sadece yaprak değil, her seviyede arama yapılır.

Neden işe yarar?
   "2020'den 2023'e kadar genel trendler neler?" gibi
   ÇOK GENİŞ KAPSAMLI sorularda tek bir chunk yetmez.
   Özet ağacında üst düzey node'lar bu tür soruları cevaplar.

Nasıl yapılır?
   1. Chunk'lara böl (yapraklar)
   2. Her X chunk'ı LLM ile özetle (ara düğümler)
   3. Ara düğümleri de özetle (kök)
   4. Tüm seviyeleri ChromaDB'ye ekle
   5. Arama sırasında tüm seviyelerde ara

Basitleştirilmiş kod:
   leaves   = split_documents(docs, chunk_size=300)
   summaries = [llm.invoke("Özetle: " + " ".join(
       [l.page_content for l in leaves[i:i+5]])) for i in range(0, len(leaves), 5)]
   all_docs = leaves + summary_docs
   vectordb = Chroma.from_documents(all_docs, embeddings)

Bu projede neden uygulamadık?
   → 57 sayfalık rapor için overkill (fazla karmaşık)
   → Groq free token limiti özetleme döngüsü için yetmezdi
   → Sonuç iyileşmesi marjinal, maliyet yüksek

Ne zaman kullan? (Kesinlikle gerekli olduğu durumlar)
   → 500+ sayfalık belgeler (kanun kitabı, ansiklopedi vb.)
   → "Genel özet nedir?" tarzı geniş kapsamlı sorular çoksa
   → Hiyerarşik bir veri yapısı varsa (bölüm > konu > detay)


─────────────────────────────────────────────────────────────────
MULTI-MODAL RAG — Görselleri de İndeksle
─────────────────────────────────────────────────────────────────
Nedir?
   PDF içindeki metin değil, aynı zamanda GRAFİKLER, TABLOLAR,
   DİYAGRAMLAR ve GÖRSELLER de aranabilir ve LLM'e gönderilebilir.

İki yaklaşım var:

YAKLAŞıM A — Görselleri metne çevir (bu projede yaptık):
   Sayfa → PNG → Vision API → metin → ChromaDB
   Artı  : Mevcut RAG pipeline değişmez.
   Eksi  : Görseldeki bağlamın bir kısmı kaybolur.

YAKLAŞıM B — Görselleri embedding'e çevir (gerçek multimodal):
   CLIP, LLaVA veya GPT-4V ile görsel → embedding vektörü
   Bu vektörü ChromaDB'ye kaydet.
   Soru gelince: metin + görsel embedding'leri birlikte ara.

   from langchain_community.embeddings import OpenCLIPEmbeddings
   image_embedding = OpenCLIPEmbeddings()
   # Görsel ve metin aynı vektör uzayında

Ne zaman kullan?
   → PDF'lerde tablo veya grafik yoğunsa ve metin yetersizse
   → "Bu grafik neyi gösteriyor?" tarzı sorular gelecekse
   → Tıbbi görüntüler, mühendislik şemaları, haritalar

Bu projede nasıl uyguladık?
   Yaklaşım A: Groq Vision ile her sayfayı metin haline getirdik.
   Tablolar çoğunlukla başarılı, karmaşık grafikler kısmi başarı.

Gerçek Multi-modal için ne gerekir?
   pip install open_clip_torch pillow
   Model: GPT-4V, Claude-3, LLaVA, Idefics
   ChromaDB multi-modal collection


─────────────────────────────────────────────────────────────────
AGENTIC RAG — Çok Kaynaklı Akıllı Routing
─────────────────────────────────────────────────────────────────
Nedir?
   Tek bir vektör tabanına değil, birden fazla veri kaynağına
   akıllıca yönlendiren bir RAG sistemidir.

   Klasik RAG: Soru → PDF Vektör DB → Cevap
   Agentic RAG: Soru → Router → [PDF mi? Web mi? SQL mi?] → Cevap

Kaynaklar şunlar olabilir:
   - Vektör DB (PDF, doküman)
   - Web arama (Tavily, SerpAPI)
   - SQL veritabanı
   - API endpoint
   - E-posta / Slack geçmişi

Nasıl çalışır? (LangGraph ile):

   def route_question(state):
       # LLM'e sor: "Bu soruyu nerede aramalıyım?"
       prompt = (
           "Soru: " + state["question"] + "\n"
           "Secenekler: pdf, web, sql\n"
           "Sadece bir kelime yaz."
       )
       decision = llm.invoke(prompt)
       return decision.content.strip()

   workflow.add_conditional_edges(START, route_question, {
       "pdf": "retrieve_pdf",
       "web": "search_web",
       "sql": "query_database"
   })

Neden bu projede yapmadık?
   → Tek kaynak (PDF raporlar) yeterliydi
   → Web araması Groq rate limitini hızlandırırdı

Ne zaman şart olur?
   → Güncel bilgi + geçmiş bilgi karışık sorular geliyorsa
     ("Bugünkü döviz kuru ve 2023 raporu baz alarak...")
   → Birden fazla şirket/kaynak varsa
   → Kullanıcı sorgusuna göre kaynak seçimi gerekiyorsa

Araçlar:
   Tavily  → web arama (LangChain ile entegre, ücretsiz plan var)
   DuckDuckGo Search → ücretsiz web arama
   SQLDatabaseChain → SQL sorguları
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
