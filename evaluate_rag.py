"""
LangSmith Dataset & Experiment
-------------------------------
Bu script:
1. LangSmith'te bir dataset olusturur (soru + beklenen cevap ciftleri)
2. RAG sistemini bu dataset uzerinde calistirir
3. LLM tabanli bir evaluator ile cevaplari skorlar (0 veya 1)
4. Sonuclari LangSmith'te Experiments sekmesinde gosterir
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from rag_graph import get_answer, llm
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
client = Client()

DATASET_NAME = "THY RAG Test Seti"

# Soru + beklenen cevap ciftleri
ornekler = [
    {
        "inputs":  {"question": "2022 yilinda hangi ucak tipleri icin bakim yapildi?"},
        "outputs": {"answer": "A350, Boeing 787 Dreamliner ve Boeing 777-300ER"}
    },
    {
        "inputs":  {"question": "2022 raporunda hayata gecirilen projeler nelerdir?"},
        "outputs": {"answer": "Tiedown Certification, Door Hinge Latch Assembly, Escutcheon, Inner Pane, Waste Container projeleri"}
    },
    {
        "inputs":  {"question": "2023 yilindaki yurt ici musteriler kimlerdir?"},
        "outputs": {"answer": "ACM Havayollari, THY Destek Hizmetleri, Pegasus gibi havayolu sirketleri"}
    },
    {
        "inputs":  {"question": "2020 yilinda net kar ne kadar?"},
        "outputs": {"answer": "85.455.249 TL"}
    },
    {
        "inputs":  {"question": "2021 yilinda net kar ne kadar?"},
        "outputs": {"answer": "309.399.776 TL"}
    },
]

# Dataset olustur (varsa sil ve yeniden olustur)
print(f"Dataset olusturuluyor: '{DATASET_NAME}'...")
existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
if existing:
    client.delete_dataset(dataset_id=existing[0].id)
    print("  Eski dataset silindi.")

dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description="THY Teknik faaliyet raporlari uzerinde RAG sistemi test seti"
)
client.create_examples(
    inputs=[e["inputs"] for e in ornekler],
    outputs=[e["outputs"] for e in ornekler],
    dataset_id=dataset.id
)
print(f"  {len(ornekler)} ornek eklendi.")


# RAG sistemini her soru icin calistiran fonksiyon
def rag_pipeline(inputs: dict) -> dict:
    result = get_answer(inputs["question"])
    return {"answer": result["result"]}


# LLM tabanli evaluator: uretilen cevap beklenen cevabi karsilayor mu?
def correctness_evaluator(run, example) -> dict:
    prediction = run.outputs.get("answer", "")
    reference  = example.outputs.get("answer", "")
    question   = example.inputs.get("question", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen bir RAG sistemi degerlendiricisin.
Kullanicinin sorusuna verilen cevap, beklenen cevaptaki bilgiyi iceriyor mu?
Evet ise 'correct', hayir ise 'incorrect' yaz. Sadece bu iki kelimeden birini yaz."""),
        ("human", "Soru: {question}\nBeklenen: {reference}\nUretilen: {prediction}")
    ])
    chain = prompt | llm
    try:
        result = chain.invoke({
            "question": question,
            "reference": reference,
            "prediction": prediction
        })
        verdict = result.content.strip().lower()
        score = 1 if "correct" in verdict and "incorrect" not in verdict else 0
        return {"key": "correctness", "score": score, "comment": verdict}
    except Exception as e:
        return {"key": "correctness", "score": 0, "comment": f"Hata: {e}"}


print("\nExperiment basliyor (5 soru, her biri LLM ile degerlendirilecek)...")
results = evaluate(
    rag_pipeline,
    data=DATASET_NAME,
    evaluators=[correctness_evaluator],
    experiment_prefix="thy-rag-v1",
    metadata={"model": "llama-3.3-70b-versatile", "retriever": "chroma-k10"}
)

print("\n=== SONUCLAR ===")
for r in results:
    q   = r["example"].inputs["question"][:50]
    ans = r["run"].outputs.get("answer", "")[:60]
    score = r["evaluation_results"]["results"][0].score if r["evaluation_results"]["results"] else "?"
    print(f"  {'OK' if score == 1 else 'XX'}  {q}")
    print(f"       Cevap: {ans}")

print("\nLangSmith'te goruntule:")
print("  smith.langchain.com → thy_rag → Datasets & Experiments")
