import os
import fitz
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyCtN5qIj5G8R1bVS9dFRZvixj5Fy8h33XE"))
model = genai.GenerativeModel("gemini-1.5-flash")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

def create_faiss_index(texts):
    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def summarize(text):
    korean_prompt = f"다음 내용을 한국어로 요약해 주세요:\n\n{text[:4000]}"
    english_prompt = f"Please summarize the following in English:\n\n{text[:4000]}"
    kr = model.generate_content(korean_prompt).text.strip()
    en = model.generate_content(english_prompt).text.strip()
    return kr, en

def answer_question(text, question):
    prompt = f"""다음은 책의 일부입니다:

{text[:4000]}

질문: {question}
답변:"""
    return model.generate_content(prompt).text.strip()

def answer_question_with_vector(query, texts, embeddings):
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    top_texts = [texts[i] for i in top_indices]
    context = "\n\n".join(top_texts)
    return answer_question(context, query)

# Streamlit UI
st.set_page_config(page_title="📄 PDF 범위 요약 및 질의응답", layout="wide")
st.title("📄 PDF 구간 요약 및 질문 챗봇")

uploaded_file = st.file_uploader("📤 PDF 파일 업로드", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    all_pages = extract_text_by_page(pdf_path)
    total_pages = len(all_pages)
    st.success(f"총 {total_pages} 페이지를 가진 PDF가 업로드되었습니다.")

    start_page = st.number_input("시작 페이지 (1부터 시작)", min_value=1, max_value=total_pages, value=1)
    end_page = st.number_input("끝 페이지", min_value=1, max_value=total_pages, value=min(5, total_pages))

    if start_page <= end_page:
        selected_texts = all_pages[start_page - 1:end_page]
        section_text = "\n".join(selected_texts)

        st.subheader("4. 구간 요약")
        if st.button("📘 요약 시작"):
            with st.spinner("요약 중..."):
                kr_sum, en_sum = summarize(section_text)
                st.markdown("### 🇰🇷 한국어 요약")
                st.write(kr_sum)
                st.markdown("### 🇺🇸 영어 요약")
                st.write(en_sum)

        st.subheader("5. 구간 내용에서 질문")
        question1 = st.text_input("질문1 입력:")
        if question1:
            with st.spinner("답변 생성 중..."):
                answer1 = answer_question(section_text, question1)
                st.markdown("### 🤖 답변 1")
                st.write(answer1)

        st.subheader("6. 전체 내용에서 질문")
        question2 = st.text_input("질문2 입력:")
        if question2:
            with st.spinner("벡터 인덱스 생성 중..."):
                index, embeddings = create_faiss_index(all_pages)
                with st.spinner("답변 생성 중..."):
                    answer2 = answer_question_with_vector(question2, all_pages, embeddings)
                    st.markdown("### 🤖 답변 2")
                    st.write(answer2)
    else:
        st.warning("시작 페이지는 끝 페이지보다 작거나 같아야 합니다.")
