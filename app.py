import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from io import BytesIO

st.title("📄 AI PDF OCR + Semantic Search System")

# 1️⃣ Optional file upload
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# 2️⃣ Page selection
pages_input = st.text_input("Enter page range (e.g., 'all' or '1-5')", value="all")

# 3️⃣ OCR & Chunking
chunks = []
metadata = []

for file in uploaded_files:
    images = convert_from_bytes(file.read(), dpi=300)
    if pages_input.lower() != "all":
        first, last = map(int, pages_input.split("-"))
        images = images[first-1:last]
    text = ""
    for i, img in enumerate(images, start=1):
        text += pytesseract.image_to_string(img, lang="ara+eng")
    # Split into chunks
    chunk_size = 500
    for j in range(0, len(text), chunk_size):
        chunks.append(text[j:j+chunk_size])
        metadata.append((file.name, i))  # store filename + page

st.write(f"Processed {len(chunks)} text chunks")

# 4️⃣ Build Embeddings & FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 5️⃣ Chat-like query interface
query = st.text_input("Ask a question about your PDFs:")
if query:
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=5)
    for i in I[0]:
        file_name, page_num = metadata[i]
        st.write(f"PDF: {file_name}, Page: {page_num}\n{chunks[i]}\n---")

# 6️⃣ Export to Excel
if st.button("Export all chunks to Excel"):
    df = pd.DataFrame({"File Name": [m[0] for m in metadata], "Page": [m[1] for m in metadata], "Text": chunks})
    excel_path = "OCR_results.xlsx"
    df.to_excel(excel_path, index=False)
    st.download_button("Download Excel", open(excel_path, "rb"), file_name="OCR_results.xlsx")
