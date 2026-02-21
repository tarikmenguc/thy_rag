# Turkish Technic RAG Assistant ✈️

An **Adaptive RAG** application that answers questions based on the 2020–2023 annual reports of **Turkish Technic (Türk Hava Yolları Teknik A.Ş.)**.

## 🎯 Project Goal & Learnings

Built to learn and demonstrate modern Generative AI and RAG techniques.

**Topics Covered:**
- **RAG Architecture** — grounding LLM responses with retrieved document context
- **Vector Database** — storing and querying text chunk embeddings with ChromaDB
- **LangGraph** — stateful graph-based Adaptive RAG with nodes, edges, and loops
- **Vision AI** — extracting text from scanned PDFs using Groq Vision
- **Metadata Filtering** — year-based document filtering in ChromaDB
- **Conversational Memory** — context-aware follow-up questions using chat history
- **Re-Ranking** — selecting the most relevant documents with FlashRank

## 🏗️ Architecture

```
User Question
      │
      ▼
 [retrieve]  ← Fetches relevant documents from ChromaDB (with year filter)
      │
      ▼
 [generate]  ← Generates an answer using Groq LLM
      │
      ▼
[grade_answer] ← Evaluates the answer: useful or not_useful?
      │
      ├── useful     → Return to user ✅
      └── not_useful → Back to [retrieve] 🔄 (max 2 retries)
```

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **LangGraph** | Adaptive RAG flow control (graph + conditional loop) |
| **LangChain** | Prompt templates, retrievers, chain composition |
| **Groq** (`llama-3.3-70b-versatile`) | Main language model |
| **Groq** (`llama-4-scout-17b`) | Vision model — OCR for scanned PDFs |
| **ChromaDB** | Vector database (1,078 chunks) |
| **HuggingFace** (`paraphrase-multilingual-MiniLM-L12-v2`) | Multilingual embeddings |
| **Streamlit** | Chat interface |
| **PyMuPDF** | PDF → PNG conversion for vision OCR |
| **FlashRank** | Re-ranking retrieved documents |

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/tarikmenguc/thy_rag.git
cd thy_rag

# 2. Install dependencies
pip install langchain langchain-groq langchain-huggingface langchain-chroma
pip install chromadb pypdf sentence-transformers python-dotenv
pip install streamlit pymupdf pillow flashrank langgraph

# 3. Create a .env file
echo GROQ_API_KEY=gsk_... > .env

# 4. Ingest 2020-2021 PDFs (text-based)
python main.py

# 5. Ingest 2022-2023 PDFs (scanned → Groq Vision OCR)
python ingest_scanned_pdfs.py

# 6. Run the application
streamlit run app.py
```

## 📂 Project Structure

```
thy_rag/
├── main.py                  # 2020-2021 PDF ingestion pipeline
├── ingest_scanned_pdfs.py   # 2022-2023 scanned PDF ingestion (Groq Vision OCR)
├── rag_graph.py             # LangGraph Adaptive RAG engine
├── app.py                   # Streamlit chat interface
├── project_journey.py       # Full project development journey (documentation)
├── .env                     # API keys
├── data/                    # Annual report PDFs (2020–2023)
└── chroma_db_thy/           # Local vector database
```

## 💬 Usage

Open the app, select the years to search in the left sidebar, and ask your questions:

- *"Which aircraft types were maintained in 2022?"*
- *"What is the staff count and title breakdown for 2023?"*
- *"What was the net profit in 2021? And what about 2020?"* ← memory support

## ⚠️ Notes

- **Free Tier Limit:** Groq's free tier allows 100K tokens/day. Each question triggers 2 LLM calls (generate + grade).
- **2022–2023 Content:** These reports are operational, not financial — they cover technical activities, certifications, and client information.
- **Naming Conflict:** Do not name any file `langgraph.py` — it conflicts with the `langgraph` Python package.
