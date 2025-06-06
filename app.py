import os
import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

# 1. 환경 변수 로드 및 Gemini 설정
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyCtN5qIj5G8R1bVS9dFRZvixj5Fy8h33XE"))
model = genai.GenerativeModel("gemini-1.5-flash")

# 2. SentenceTransformer 초기화 (CPU 기본 사용)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. PDF 텍스트 추출
def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

# 4. 벡터 생성
def create_embeddings(texts):
    return embedder.encode(texts)

# 5. 유사한 페이지 기반 질문 응답
def answer_question_with_vector(query, texts, embeddings):
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    top_texts = [texts[i] for i in top_indices]
    context = "\n\n".join(top_texts)
    return answer_question(context, query)

# 6. Gemini 요약
def summarize(text):
    kr_prompt = f"다음 내용을 한국어로 요약해 주세요:\n\n{text[:4000]}"
    en_prompt = f"Please summarize the following in English:\n\n{text[:4000]}"
    kr = model.generate_content(kr_prompt).text.strip()
    en = model.generate_content(en_prompt).text.strip()
    return kr, en

# 7. Gemini QA
def answer_question(context, question):
    prompt = f"""다음은 문서의 일부입니다:

{context[:4000]}

질문: {question}
답변:"""
    return model.generate_content(prompt).text.strip()

# 8. Streamlit UI
st.set_page_config(page_title="📄 PDF 구간 요약 및 Q&A", layout="wide")
st.title("📄 PDF 구간 요약 & 질문 챗봇")

uploaded_file = st.file_uploader("📤 PDF 파일 업로드", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    all_pages = extract_text_by_page(pdf_path)
    total_pages = len(all_pages)
    st.success(f"총 {total_pages}페이지 PDF가 업로드되었습니다.")

    start_page = st.number_input("시작 페이지 (1부터)", min_value=1, max_value=total_pages, value=1)
    end_page = st.number_input("끝 페이지", min_value=1, max_value=total_pages, value=start_page)

    if start_page > end_page:
        st.warning("⚠ 시작 페이지는 끝 페이지보다 작거나 같아야 합니다.")
    else:
        selected_texts = all_pages[start_page - 1:end_page]
        section_text = "\n".join(selected_texts)

        st.subheader("📘 선택 구간 요약")
        if st.button("요약 시작"):
            with st.spinner("요약 중..."):
                kr_sum, en_sum = summarize(section_text)
                st.markdown("### 🇰🇷 한국어 요약")
                st.write(kr_sum)
                st.markdown("### 🇺🇸 영어 요약")
                st.write(en_sum)

        st.subheader("❓ 선택 구간에 대해 질문")
        question = st.text_input("질문을 입력하세요:")
        if question:
            with st.spinner("벡터 인덱스 생성 및 답변 중..."):
                embeddings = create_embeddings(selected_texts)
                answer = answer_question_with_vector(question, selected_texts, embeddings)
                st.markdown("### 🤖 답변")
                st.write(answer)
