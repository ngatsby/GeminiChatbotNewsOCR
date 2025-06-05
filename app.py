import os
import fitz
import pytesseract
import streamlit as st
from PIL import Image
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

# 환경 변수에서 Gemini API Key 불러오기
load_dotenv()
genai.configure(api_key=os.getenv("API key"))  # .env에 GOOGLE_API_KEY 설정
model = genai.GenerativeModel("gemini-1.5-flash")

# OCR로 PDF 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="kor+eng", config="--psm 1")
        full_text += text + "\n"
    return full_text.strip()

# 한글 및 영어 요약
def summarize_in_korean_and_english(article_text):
    prompt_kr = f"다음 기사를 한국어로 간결하게 요약해 주세요:\n\n{article_text}"
    prompt_en = f"Please summarize the following article in English:\n\n{article_text}"

    response_kr = model.generate_content(prompt_kr).text.strip()
    response_en = model.generate_content(prompt_en).text.strip()

    return response_kr, response_en

# Streamlit UI
st.set_page_config("신문 요약 챗봇", layout="centered")
st.title("📰 신문 기사 요약 및 질의응답 챗봇")

uploaded_file = st.file_uploader("📄 OCR이 필요한 신문 PDF 업로드", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    with st.spinner("🔍 OCR로 텍스트 인식 중..."):
        article_text = extract_text_from_pdf(tmp_pdf_path)

    with st.spinner("✍️ Gemini로 요약 중..."):
        summary_kr, summary_en = summarize_in_korean_and_english(article_text)

    st.markdown("### 📑 한국어 요약")
    st.write(summary_kr)

    st.markdown("### 📑 English Summary")
    st.write(summary_en)

    # 추가 질문
    user_question = st.text_input("❓ 기사에 대해 질문해 보세요:")
    if user_question:
        with st.spinner("🤖 답변 생성 중..."):
            q_prompt = f"다음은 기사입니다:\n\n{article_text}\n\n사용자 질문: {user_question}\n\n답변:"
            answer = model.generate_content(q_prompt).text.strip()
            st.markdown("### 🤖 답변")
            st.write(answer)
