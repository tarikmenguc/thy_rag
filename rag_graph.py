import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectordb = Chroma(
    persist_directory="./chroma_db_thy",
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})


class RAGState(TypedDict):
    question: str
    hypothesis: str      # HyDE: soruden uretilen hayali cevap
    documents: list
    answer: str
    grade: str
    retries: int
    year_filter: list


def generate_hypothesis(state: RAGState):
    """
    HyDE: Soruya hayali bir cevap uret.
    Bu cevap gercek olmayabilir ama semantik olarak
    gercek dokumanlara daha yakin olacak.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen bir THY Teknik uzmanissin.
    Asagidaki soruya kisa ve net bir cevap yaz.
    Cevap hayali olabilir ama gercekci olmali.
    1-2 cumle yeterli."""),
        ("human", "{question}")
    ])
    chain = prompt | llm
    try:
        result = chain.invoke({"question": state["question"]})
        hypothesis = result.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            hypothesis = state["question"]  # rate limit: soruyu oldugu gibi kullan
        else:
            raise
    return {"hypothesis": hypothesis}

def retrieve(state: RAGState):
    year_filter = state.get("year_filter", [])
    search_query = state.get("hypothesis") or state["question"]
    if year_filter:
        docs = vectordb.similarity_search(
            search_query,
            k=10,
            filter={"year": {"$in": year_filter}}
        )
    else:
        docs = vectordb.similarity_search(search_query, k=10)
    return {"documents": docs}


def generate(state: RAGState):
    context = "\n\n".join(d.page_content for d in state["documents"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Asagidaki baglamdaki bilgileri kullanarak soruyu cevapla. Cevabi bilmiyorsan 'Bilmiyorum' de.\n\nBaglam:\n{context}"),
        ("human", "{question}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"context": context, "question": state["question"]})
        answer = response.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            answer = "API gunluk token limiti doldu. Lutfen yarin tekrar deneyin."
        else:
            raise
    return {"answer": answer, "retries": state["retries"] + 1}


def grade_answer(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Kullanicinin sorusuna verilen cevabi degerlendir. Cevap somut bilgi iceriyorsa 'useful', belirsiz veya 'bilmiyorum' iceriyorsa 'not_useful' yaz. Sadece bu iki kelimeden birini yaz."),
        ("human", "Soru: {question}\nCevap: {answer}")
    ])
    try:
        chain = prompt | llm
        result = chain.invoke({"question": state["question"], "answer": state["answer"]})
        grade = result.content.strip().lower()
        if "useful" in grade and "not" not in grade:
            return {"grade": "useful"}
        return {"grade": "not_useful"}
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return {"grade": "useful"}
        raise


def decide_next(state: RAGState) -> str:
    if state["grade"] == "useful":
        return "end"
    if state["retries"] >= 2:
        return "end"
    return "retrieve"


workflow = StateGraph(RAGState)
workflow.add_node("generate_hypothesis", generate_hypothesis)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade_answer", grade_answer)
workflow.add_edge(START, "generate_hypothesis")   # HyDE ilk adim
workflow.add_edge("generate_hypothesis", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_answer")
workflow.add_conditional_edges("grade_answer", decide_next, {"end": END, "retrieve": "retrieve"})
app = workflow.compile()


def get_answer(query: str, chat_history=[], year_filter=None) -> dict:
    result = app.invoke({
        "question": query,
        "hypothesis": "",
        "documents": [],
        "answer": "",
        "grade": "",
        "retries": 0,
        "year_filter": year_filter or []
    })
    return {
        "result": result["answer"],
        "source_documents": result["documents"]
    }


if __name__ == "__main__":
    soru = sys.argv[1] if len(sys.argv) > 1 else "2022 yilinda hangi projeler yapildi?"
    print(f"Soru: {soru}\n" + "-" * 50)
    cevap = get_answer(soru)
    print(f"Cevap: {cevap['result']}")
    for doc in cevap["source_documents"][:3]:
        print(f"  - Yil {doc.metadata.get('year')} | Sayfa {doc.metadata.get('page')}")
