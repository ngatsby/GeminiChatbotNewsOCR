import os
import fitz
import streamlit as st
import faiss
import numpy as np
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyCtN5qIj5G8R1bVS9dFRZvixj5Fy8h33XE"))
model = genai.GenerativeModel("gemini-1.5-flash")
embed_model = genai.GenerativeModel("embedding-001")

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

def get_gemini_embedding(text):
    try:
        res = embed_model.embed_content(
            content=text,
            task_type="retrieval_document"
        )
        return np.array(res['embedding'], dtype='float32')
    except Exception as e:
        st.error(f"❌ 임베딩 실패: {e}")
        return None

def create_faiss_index(texts):
    embeddings = [get_gemini_embedding(t) for t in texts]
    valid = [(e, t) for e, t in zip(embeddings, texts) if e is not None]
    if not valid:
        st.error("❌ 유효한 임베딩이 없습니다.")
        return None, None
    emb_array = np.array([v[0] for v in valid])
    texts = [v[1] for v in valid]
    index = faiss.IndexFlatL2(emb_array.shape[1])
    index.add(emb_array)
    return index, emb_array, texts

def summarize(text):
    kr_prompt = f"다음 내용을 한국어로 요약해 주세요:\n\n{text[:4000]}"
    en_prompt = f"Please summarize the following in English:\n\n{text[:4000]}"
    kr = model.generate_content(kr_prompt).text.strip()
    en = model.generate_content(en_prompt).text.strip()
    return kr, en

def answer_question(text, question):
    prompt = f"""다음은 책의 일부입니다:

{text[:4000]}

질문: {question}
답변:"""
    return model.generate_content(prompt).text.strip()

def answer_question_with_vector(query, faiss_index, texts, emb_array):
    q_emb = get_gemini_embedding(query)
    if q_emb is None:
        return "❌ 질문 임베딩에 실패했습니다."
    _, I = faiss_index.search(np.array([q_emb]), k=3)
    top_texts = [texts[i] for i in I[0]]
    context = "\n\n".join(top_texts)
    return answer_question(context, query)

# Streamlit UI
st.set_page_config(page_title="📄 PDF 범위 요약 및 질의응답", layout="wide")
st.title("📄 PDF 구간 요약 및 질문 챗봇 (Gemini Embedding)")

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
                faiss_index, emb_array, valid_texts = create_faiss_index(all_pages)
                if faiss_index:
                    with st.spinner("답변 생성 중..."):
                        answer2 = answer_question_with_vector(question2, faiss_index, valid_texts, emb_array)
                        st.markdown("### 🤖 답변 2")
                        st.write(answer2)
    else:
        st.warning("시작 페이지는 끝 페이지보다 작거나 같아야 합니다.")
