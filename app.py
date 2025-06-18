import streamlit as st

st.set_page_config(page_title="📄 잡지/책 PDF 챗봇", layout="wide")

import tempfile
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from gtts import gTTS  # TTS 라이브러리 추가
import io

# --- Configuration ---
GEMINI_API_KEY = st.secrets.get("Key")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in Streamlit secrets. Please add 'Key' to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Global Models (Load once for efficiency) ---
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("models/gemini-1.5-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_gemini_model()
embedder = load_embedding_model()

# --- Helper Functions ---
def extract_pages_from_pdf(pdf_file_path):
    try:
        reader = PdfReader(pdf_file_path)
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []

def create_embeddings(texts):
    if not texts:
        return np.array([])
    return embedder.encode(texts)

def build_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ask_gemini(question, context):
    prompt = f"""
    다음 기사 내용 기반으로 질문에 답해주세요.
    ---
    {context}
    ---
    질문: {question}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "I'm sorry, I couldn't generate a response."

def play_tts(text, lang='ko'):
    try:
        tts = gTTS(text, lang=lang)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        st.audio(mp3_fp.getvalue(), format="audio/mp3")
    except Exception as e:
        st.error(f"TTS 변환 중 오류가 발생했습니다: {e}")

# --- Streamlit UI ---
st.title("📄 잡지/책 PDF 챗봇")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "all_pages" not in st.session_state:
    st.session_state.all_pages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "selected_texts" not in st.session_state:
    st.session_state.selected_texts = []

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    if not st.session_state.pdf_processed or st.session_state.get("uploaded_file_name", "") != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("PDF를 처리 중입니다..."):
            st.session_state.all_pages = extract_pages_from_pdf(tmp_path)
            st.session_state.pdf_processed = True
            st.success("PDF 업로드 및 페이지 추출 완료")

        try:
            os.remove(tmp_path)
        except OSError as e:
            st.warning(f"Warning: Could not remove temporary file {tmp_path}: {e}")

if st.session_state.pdf_processed and st.session_state.all_pages:
    total_pages = len(st.session_state.all_pages)
    st.write(f"총 {total_pages} 페이지가 추출되었습니다.")

    start_page = st.number_input("시작 페이지 (1부터 시작)", min_value=1, max_value=total_pages, value=1, key="start_page_input")
    end_page = st.number_input("끝 페이지", min_value=start_page, max_value=total_pages, value=min(start_page + 4, total_pages), key="end_page_input")

    if st.button("선택한 구간 분석 시작", key="analyze_button"):
        with st.spinner("선택한 구간을 벡터화 중입니다..."):
            st.session_state.selected_texts = st.session_state.all_pages[start_page - 1:end_page]
            if st.session_state.selected_texts:
                embeddings = create_embeddings(st.session_state.selected_texts)
                st.session_state.index = build_faiss_index(np.array(embeddings))
                if st.session_state.index:
                    st.success("해당 구간 벡터화 완료! 이제 질문할 수 있습니다.")
                else:
                    st.warning("선택된 페이지에서 텍스트를 추출할 수 없거나 임베딩을 생성할 수 없습니다.")
            else:
                st.warning("선택된 페이지 범위에 텍스트가 없습니다.")

    if st.session_state.index:
        question = st.text_input("질문을 입력하세요", key="question_input")

        if question:
            with st.spinner("응답 생성 중..."):
                try:
                    question_embedding = embedder.encode([question])
                    D, I = st.session_state.index.search(np.array(question_embedding), k=3)
                    matched_texts = [st.session_state.selected_texts[i] for i in I[0] if i < len(st.session_state.selected_texts)]
                    
                    if matched_texts:
                        context = "\n".join(matched_texts)
                        answer = ask_gemini(question, context)
                        st.markdown("#### 🧠 Gemini의 답변")
                        st.write(answer)
                        # TTS 버튼 추가
                        if st.button("🔊 답변 듣기"):
                            # 답변이 영어면 lang='en', 한글이면 lang='ko'로 자동 처리(간단히 한글 포함 여부로 구분)
                            lang = 'ko' if any('\uac00' <= c <= '\ud7a3' for c in answer) else 'en'
                            play_tts(answer, lang=lang)
                    else:
                        st.info("관련된 텍스트를 찾을 수 없습니다. 질문을 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"질의응답 중 오류가 발생했습니다: {e}")
