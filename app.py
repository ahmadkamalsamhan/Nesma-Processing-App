import streamlit as st
import json
from io import BytesIO
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests

st.set_page_config(page_title="PDF OCR + AI Search", layout="wide")
st.title("📄 Nesma PDF OCR + AI Semantic Search System")

# --------------------------
# Service Account Setup via Streamlit Secrets
# --------------------------
service_account_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
SCOPES = ["https://www.googleapis.com/auth/drive"]

credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)

# --------------------------
# PDF Selection
# --------------------------
st.subheader("PDF Sources")

# 1️⃣ Google Drive Folder ID input
folder_id = st.text_input("Paste Google Drive folder ID (optional)")

# 2️⃣ Local PDF upload
uploaded_files = st.file_uploader(
    "Upload PDFs (optional)", type="pdf", accept_multiple_files=True
)

# Page range
pages_input = st.text_input(
    "Enter page range for PDFs (all or 1-5)", value="all"
)

# --------------------------
# Helper Functions
# --------------------------
def list_pdfs_in_folder(folder_id):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id, name)"
    ).execute()
    return results.get("files", [])

def download_pdf(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    return r.content

def extract_text_from_pdf(file_bytes, filename, page_range="all"):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if page_range.lower() != "all":
        first, last = map(int, page_range.split("-"))
        pages = range(first-1, last)
    else:
        pages = range(len(doc))
    full_text = ""
    local_chunks = []
    local_metadata = []
    for i in pages:
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="ara+eng")
        full_text += text + "\n"
        # Chunk text
        chunk_size = 500
        for j in range(0, len(text), chunk_size):
            local_chunks.append(text[j:j+chunk_size])
            local_metadata.append((filename, i+1))
    return full_text, local_chunks, local_metadata

# --------------------------
# Process PDFs
# --------------------------
chunks = []
metadata = []

if folder_id:
    st.info("Downloading PDFs from Google Drive folder...")
    files = list_pdfs_in_folder(folder_id)
    for f in files:
        pdf_bytes = download_pdf(f["id"])
        _, local_chunks, local_metadata = extract_text_from_pdf(pdf_bytes, f["name"], pages_input)
        chunks.extend(local_chunks)
        metadata.extend(local_metadata)

if uploaded_files:
    st.info("Processing uploaded PDFs...")
    for f in uploaded_files:
        _, local_chunks, local_metadata = extract_text_from_pdf(f.read(), f.name, pages_input)
        chunks.extend(local_chunks)
        metadata.extend(local_metadata)

st.write(f"Total text chunks created: {len(chunks)}")

# --------------------------
# AI Semantic Search
# --------------------------
if chunks:
    st.info("Building AI embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    query = st.text_input("Ask a question about your PDFs:")
    if query:
        query_vec = model.encode([query])
        D, I = index.search(query_vec, k=5)
        st.success("Top relevant chunks:")
        for i in I[0]:
            file_name, page_num = metadata[i]
            st.write(f"📄 {file_name} | Page {page_num}\n{chunks[i]}\n---")

# --------------------------
# Export to Excel
# --------------------------
if st.button("Export all chunks to Excel"):
    df = pd.DataFrame({
        "File Name": [m[0] for m in metadata],
        "Page": [m[1] for m in metadata],
        "Text": chunks
    })
    excel_path = "OCR_results.xlsx"
    df.to_excel(excel_path, index=False)
    st.download_button("Download Excel", open(excel_path, "rb"), file_name="OCR_results.xlsx")
