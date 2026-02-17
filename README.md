# Turkish Technic RAG Assistant ✈️

This project is a **Retrieval-Augmented Generation (RAG)** application designed to answer questions based on the Annual Reports (2020-2023) of **Turkish Technic** (Türk Hava Yolları Teknik A.Ş.). It uses **Groq** (Llama 3) for the LLM and **ChromaDB** for vector storage, wrapped in a user-friendly **Streamlit** interface.

## 🎯 Purpose & Learnings
I created this project to learn and demonstrate key concepts of Generative AI and RAG systems.

### Key Concepts Mastered:
- **RAG Architecture**: Understanding how to retrieve relevant context from external documents to ground LLM responses.
- **Vector Databases**: Using **ChromaDB** to store and query high-dimensional embeddings of text chunks.
- **Embeddings**: Utilizing `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) to convert Turkish text into semantic vectors.
- **LangChain Framework**: Orchestrating the flow between the LLM, retriever, and vector store.
- **LLM Integration**: Leveraging the speed of **Groq API** with the powerful **Llama 3** model.
- **Prompt Engineering**: Designing prompts to force the model to use *only* retrieved context and cite sources.
- **Frontend Development**: Building a chat interface with **Streamlit** for real-time interaction.

## 🛠️ Tech Stack
- **LLM**: Llama 3 (via Groq API)
- **Framework**: LangChain
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- **Frontend**: Streamlit
- **PDF Parsing**: PyPDF / PyMuPDF

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/tarikmenguc/thy_rag.git
    cd thy_rag
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment**:
    Create a `.env` file and add your Groq API Key:
    ```env
    GROQ_API_KEY=your_api_key_here
    ```

4.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
- `app.py`: The Streamlit frontend application.
- `rag.py`: The core RAG logic (retrieval, generation, citation).
- `main.py`: Script to ingest PDFs and build the Vector Database.
- `data/`: Directory containing the PDF reports.
- `chroma_db_thy/`: The local Vector Database (embeddings).

## ⚠️ Limitations
- **Data Source**: The system is trained on **Turkish Technic** reports, not the main THY A.O. reports. Questions should be specific to the technical subsidiary.
- **PDF Quality**: The 2022 and 2023 reports were found to be image-based (scanned), resulting in limited text extraction. The 2020 and 2021 reports are fully searchable.
