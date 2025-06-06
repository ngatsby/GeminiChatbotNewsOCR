import os
import fitz  # PyMuPDF
import streamlit as st
import tempfile
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# 환경변수 로딩
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyCtN5qIj5G8R1bVS9dFRZvixj5Fy8h33XE"))
model = genai.GenerativeModel("gemini-1.5-flash")

# 임베딩 모델 (CPU 전용)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# PDF 텍스트 추출
def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

# 벡터 인덱스 생성
def create_faiss_index(texts):
    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

# 요약 함수 (한국어/영어)
def summarize(text):
    korean_prompt = f"다음 내용을 한국어로 요약해 주세요:\n\n{text[:4000]}"
    english_prompt = f"Please summarize the following in English:\n\n{text[:4000]}"
    kr = model.generate_content(korean_prompt).text.strip()
    en = model.generate_content(english_prompt).text.strip()
    return kr, en

# 질문 응답 함수
def answer_question(text, question):
    prompt = f"""다음은 문서의 일부입니다:

{text[:4000]}

질문: {question}
답변:"""
    return model.generate_content(prompt).text.strip()

# Streamlit UI
st.set_page_config(page_title="📄 PDF 구간 요약 및 질문", layout="wide")
st.title("📄 PDF 구간 요약 및 질문 챗봇")

uploaded_file = st.file_uploader("📤 PDF 파일 업로드", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    all_pages = extract_text_by_page(pdf_path)
    total_pages = len(all_pages)
    st.success(f"총 {total_pages} 페이지가 로드되었습니다.")

    start_page = st.number_input("시작 페이지 (1부터 시작)", min_value=1, max_value=total_pages, value=1)
    end_page = st.number_input("끝 페이지", min_value=1, max_value=total_pages, value=start_page)

    if start_page <= end_page:
        selected_texts = all_pages[start_page - 1:end_page]
        section_text = "\n".join(selected_texts)

        st.subheader("📘 구간 요약")
        if st.button("요약 시작"):
            with st.spinner("요약 중..."):
                kr_sum, en_sum = summarize(section_text)
                st.markdown("### 🇰🇷 한국어 요약")
                st.write(kr_sum)
                st.markdown("### 🇺🇸 영어 요약")
                st.write(en_sum)

        st.subheader("💬 구간 질문")
        question = st.text_input("질문 입력:")
        if question:
            with st.spinner("질문 응답 중..."):
                index, embeddings = create_faiss_index(selected_texts)
                query_embedding = embedder.encode([question])
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                top_indices = similarities.argsort()[-3:][::-1]
                top_texts = [selected_texts[i] for i in top_indices]
                context = "\n\n".join(top_texts)
                answer = answer_question(context, question)
                st.markdown("### 🤖 답변")
                st.write(answer)
    else:
        st.warning("⚠️ 시작 페이지는 끝 페이지보다 작거나 같아야 합니다.")
