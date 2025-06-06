import streamlit as st
import tempfile
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# 환경 설정
GEMINI_API_KEY = st.secrets["AIzaSyCtN5qIj5G8R1bVS9dFRZvixj5Fy8h33XE"]
genai.configure(api_key=GEMINI_API_KEY)

# Gemini 모델
model = genai.GenerativeModel("models/gemini-pro")

# 임베딩 모델 (CPU 기본 사용)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 텍스트 쪼개기 (페이지 단위)
def extract_pages_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return [page.extract_text() or "" for page in reader.pages]

# 임베딩 생성 함수
def create_embeddings(texts):
    embeddings = embedder.encode(texts)
    return embeddings

# 벡터 검색기 생성
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 질의응답 함수
def ask_gemini(question, context):
    prompt = f"""
    다음 기사 내용 기반으로 질문에 답해주세요.
    ---
    {context}
    ---
    질문: {question}
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📄 잡지/책 PDF 챗봇")

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF 업로드 완료")
    all_pages = extract_pages_from_pdf(tmp_path)
    total_pages = len(all_pages)

    start = st.number_input("시작 페이지 (1부터 시작)", min_value=1, max_value=total_pages, value=1)
    end = st.number_input("끝 페이지", min_value=start, max_value=total_pages, value=start)

    if st.button("선택한 구간 분석 시작"):
        selected_texts = all_pages[start-1:end]
        embeddings = create_embeddings(selected_texts)
        index = build_faiss_index(np.array(embeddings))

        st.success("해당 구간 벡터화 완료! 질문해보세요.")
        question = st.text_input("질문을 입력하세요")

        if question:
            # 가장 유사한 문장 찾기
            question_embedding = embedder.encode([question])
            D, I = index.search(np.array(question_embedding), k=3)
            matched_texts = [selected_texts[i] for i in I[0] if i < len(selected_texts)]
            context = "\n".join(matched_texts)

            with st.spinner("응답 생성 중..."):
                answer = ask_gemini(question, context)
                st.markdown("#### 🧠 Gemini의 답변")
                st.write(answer)

    os.remove(tmp_path)
