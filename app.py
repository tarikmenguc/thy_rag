import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag_graph import get_answer

st.set_page_config(
    page_title="THY Faaliyet Raporu Asistani",
    page_icon="✈️",
    layout="wide"
)

st.sidebar.title("Ayarlar")
st.sidebar.markdown("---")
selected_years = st.sidebar.multiselect(
    "Analiz Edilecek Yillar",
    options=["2020", "2021", "2022", "2023"],
    default=["2020", "2021", "2022", "2023"],
    help="Sadece sectiginiz yillara ait raporlarda arama yapilir."
)
st.sidebar.markdown("---")
st.sidebar.info("Ipucu: 'Peki ya kargo?' gibi devam sorulari sorabilirsiniz.")
if st.sidebar.button("Sohbeti Temizle"):
    st.session_state.messages = []
    st.rerun()

st.title("THY Rapor Asistani (LangGraph + Adaptive RAG)")
st.markdown("**Turk Hava Yollari Teknik A.S.** faaliyet raporlari uzerinden sorularinizi cevaplar.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Kaynaklar"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

if prompt := st.chat_input("Sorunuzu buraya yazin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("LangGraph dusunuyor... (ara > uret > degerlendir)"):
            try:
                chat_history = []
                msgs = st.session_state.messages[:-1]
                current_user_msg = None
                for msg in msgs:
                    if msg["role"] == "user":
                        current_user_msg = msg["content"]
                    elif msg["role"] == "assistant" and current_user_msg:
                        chat_history.append((current_user_msg, msg["content"]))
                        current_user_msg = None

                result = get_answer(prompt, chat_history, selected_years)
                full_response = result["result"]

                citations = []
                seen = set()
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "")
                    page = doc.metadata.get("page", 0)
                    year = doc.metadata.get("year", "")
                    key = f"**{os.path.basename(source)} (Yil: {year})** - Sayfa {page}"
                    if key not in seen:
                        citations.append(key)
                        seen.add(key)

                message_placeholder.markdown(full_response)
                if citations:
                    with st.expander("Kaynaklar"):
                        for c in citations:
                            st.markdown(f"- {c}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations
                })
            except Exception as e:
                st.error(f"Hata: {e}")
