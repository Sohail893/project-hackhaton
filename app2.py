import streamlit as st
from PIL import Image
import cv2
import numpy as np
import easyocr
import openai
from fpdf import FPDF
from openai import OpenAI
import re
# Set your OpenAI API key here
#api_key = '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'  # Replace with your real key
api = "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"
#client = OpenAI(api_key=api_key)

import requests

HUGGINGFACE_API_KEY = api
st.title("🧠 AI Medical Report Analyzer")
uploaded_file = st.file_uploader("Upload Medical Report (JPG/PNG/PDF)", type=['jpg', 'png', 'pdf'])

# 1. Preprocess the uploaded image (convert to grayscale, binarize)
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('L')  # grayscale
    img_np = np.array(image)
    _, thresh = cv2.threshold(img_np, 150, 255, cv2.THRESH_BINARY)  # binarize
    return thresh

# 2. Extract text from image using OCR
def extract_text(image_array):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_array)
    extracted_text = '\n'.join([res[1] for res in results])
    return extracted_text

# 3. Structure the extracted text into test name, value, and normal range
def structure_text(ocr_text):
    # Step 1: Normalize whitespace and remove unnecessary characters
    text = ocr_text.replace('\n', ' ').replace('’', "'").replace('“', '"')
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 2: Isolate CBC section
    match = re.search(r'COMPLETE BLOOD COUNT(.*?)Digitally signed by', text, re.IGNORECASE)
    if not match:
        return []

    cbc_section = match.group(1)

    # Step 3: Pattern to capture rows like: TestName Result NormalRange Units
    row_pattern = re.compile(
        r'([A-Z#%a-z/]+)\s+(\d+\.?\d*)\s+(\d+\.?\d*[\-–]\d+\.?\d*|Up to \d+\.?\d*|[\d\.]+)\s+([a-zA-Z/\^\d%]+)',
        re.IGNORECASE
    )

    matches = row_pattern.findall(cbc_section)
    results = []

    for match in matches:
        test_name, result, normal_range, units = match
        results.append({
            'Test Name': test_name,
            'Result': result,
            'Normal Range': normal_range,
            'Units': units
        })

    return results

# 4. Explain test results using OpenAI
# def explain_test(test_name, value, normal_range):
#     prompt = f"Explain in simple language what it means if {test_name} is {value}, given the normal range is {normal_range}."
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message['content']

import requests
HUGGINGFACE_API_KEY = "your_huggingface_token_here"

def explain_test(test_name, Result, normal_range,units):
    prompt = f"Explain in simple language what it means if {test_name} is result {Result} and the units value is {units}, given the normal range is {normal_range}."

    API_URL = "https://huggingface.co/mrm8488/flan-t5-base-common_gen"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return "Could not fetch explanation (API limit or error)."




#5. Save extracted and structured data to PDF
def save_as_pdf(text, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(filename)
    return filename

# === MAIN LOGIC ===
if uploaded_file:
    st.success("✅ File uploaded successfully!")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = preprocess_image(uploaded_file)
    text = extract_text(img)

    st.subheader("📄 Extracted Text:")
    st.text(text)

    structured_data = structure_text(text)

    st.subheader("🧪 Structured Results:")
    if structured_data:
        for row in structured_data:
            st.write(row)
            explanation = explain_test(row['test'], row['sesult'], row['normal range'],row["units"])
            with st.expander(f"ℹ️ Explanation for {row['Test']}"):
                st.write(explanation)
    else:
        st.warning("❗No structured test results could be extracted.")

    if st.button("📄 Save Report to PDF"):
        filename = save_as_pdf(text)
        with open(filename, "rb") as file:
            st.download_button("⬇️ Download PDF", file, file_name=filename, mime="application/pdf")
