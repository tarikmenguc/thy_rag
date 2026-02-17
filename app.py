import streamlit as st
import os
import sys

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag import get_answer

# Sayfa Yapılandırması
st.set_page_config(
    page_title="THY Faaliyet Raporu Asistanı",
    page_icon="✈️",
    layout="centered"
)

# Başlık ve Açıklama
st.title("✈️ THY Rapor Asistanı")
st.markdown("""
Bu asistan **Türk Hava Yolları Teknik A.Ş.**'nin 2020-2023 faaliyet raporları üzerinden sorularınızı cevaplar.
""")

# Uyarılar
st.info("💡 **Not:** 2022 ve 2023 raporları resim formatında olduğu için sadece 2020 ve 2021 yılları için detaylı cevap alabilirsiniz.")

# Sohbet Geçmişi (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 Kaynaklar"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

# Kullanıcı Girişi
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan Cevabı
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Raporlar taranıyor..."):
            try:
                result = get_answer(prompt)
                full_response = result['result']
                
                # Kaynakları düzenle
                citations = []
                seen_sources = set()
                for doc in result['source_documents']:
                    source = doc.metadata.get('source', 'Bilinmiyor')
                    page = doc.metadata.get('page', 0)
                    filename = os.path.basename(source)
                    source_key = f"**{filename}** - Sayfa {page}"
                    
                    if source_key not in seen_sources:
                        citations.append(source_key)
                        seen_sources.add(source_key)
                
                message_placeholder.markdown(full_response)
                
                if citations:
                    with st.expander("📚 Kaynaklar"):
                        for citation in citations:
                            st.markdown(f"- {citation}")
                            
                # Cevabı geçmişe ekle
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "citations": citations
                })
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
