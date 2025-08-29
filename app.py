# app.py
import streamlit as st
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
import traceback
import os

st.set_page_config(page_title="Nesma PDF OCR + AI", layout="wide")
st.title("📄 Nesma PDF OCR + AI Semantic Search System")

# --------------------------
# Read and fix service account info from Streamlit secrets
# --------------------------
if "SERVICE_ACCOUNT_JSON" not in st.secrets:
    st.error(
        "Service account secret SERVICE_ACCOUNT_JSON is missing. "
        "Add it in Streamlit Cloud -> Manage App -> Secrets."
    )
    st.stop()

# st.secrets["SERVICE_ACCOUNT_JSON"] may already be a dict (if you used TOML section),
# or it may be a string (if you pasted JSON literal). Handle both cases.
sa_raw = st.secrets["SERVICE_ACCOUNT_JSON"]

# Normalize to dict
if isinstance(sa_raw, str):
    # If a raw JSON string was stored in the secret
    try:
        import json
        service_account_info = json.loads(sa_raw)
    except Exception:
        st.error("Failed to parse SERVICE_ACCOUNT_JSON (not valid JSON). Check the secret format.")
        st.stop()
elif isinstance(sa_raw, dict):
    service_account_info = dict(sa_raw)
else:
    st.error("Unsupported SERVICE_ACCOUNT_JSON format in secrets.")
    st.stop()

# Repair common private_key formatting issues:
private_key = service_account_info.get("private_key")
if private_key:
    # If user pasted with escaped backslash-n sequences (\\n), replace them with real newlines:
    if "\\n" in private_key:
        private_key = private_key.replace("\\n", "\n")
    # If user added surrounding quotes accidentally, strip them:
    if private_key.startswith('"') and private_key.endswith('"'):
        private_key = private_key[1:-1]
    # Ensure it ends with newline
    if not private_key.endswith("\n"):
        private_key = private_key + "\n"
    # Put repaired key back
    service_account_info["private_key"] = private_key
else:
    st.error("service_account_info does not contain 'private_key'. Please set SERVICE_ACCOUNT_JSON correctly.")
    st.stop()

# Create credentials safely with error handling
SCOPES = ["https://www.googleapis.com/auth/drive"]
try:
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)
except Exception as e:
    st.error("Failed to build service account credentials. See details below.")
    st.text(traceback.format_exc())
    st.stop()

# --------------------------
# UI: PDF sources
# --------------------------
st.subheader("PDF Sources")

# Google Drive folder (folder ID)
folder_id = st.text_input("Paste Google Drive folder ID (optional). Example: 1aBcD...")

# Allow user to paste a single Google Drive file link too
drive_link = st.text_input("Or paste a Google Drive file link (optional)")

# Local file upload
uploaded_files = st.file_uploader("Upload PDFs (optional)", type="pdf", accept_multiple_files=True)

# Page selection
pages_input = st.text_input("Enter page range for PDFs (e.g., 'all' or '1-5')", value="all")

# Option: chunk size and number of search results
chunk_size = st.number_input("Chunk size (characters)", value=500, min_value=100, max_value=2000, step=100)
top_k = st.number_input("Top K results for query", value=5, min_value=1, max_value=20, step=1)

# --------------------------
# Helper functions
# --------------------------
def list_pdfs_in_folder(folder_id):
    try:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            fields="files(id, name)",
            pageSize=1000
        ).execute()
        return results.get("files", [])
    except Exception:
        st.error("Failed to list files in folder. Check folder ID and that service account has access.")
        st.text(traceback.format_exc())
        return []

def download_pdf(file_id):
    # Works for files shared with the service account or shared publicly
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Failed to download file id {file_id} (status {r.status_code}).")
        return None
    return r.content

def extract_text_from_pdf_bytes(file_bytes, filename, page_range="all", chunk_size_local=500):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        st.error(f"Cannot open PDF file {filename}. It may be corrupted.")
        return "", [], []
    if page_range.lower() != "all":
        try:
            first, last = map(int, page_range.split("-"))
            pages = range(max(0, first-1), min(len(doc), last))
        except Exception:
            pages = range(len(doc))
    else:
        pages = range(len(doc))
    full_text = ""
    local_chunks = []
    local_metadata = []
    for i in pages:
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # run OCR
        try:
            text = pytesseract.image_to_string(img, lang="ara+eng")
        except Exception:
            text = pytesseract.image_to_string(img)  # fallback
        full_text += text + "\n"
        # split into chunks
        for j in range(0, len(text), chunk_size_local):
            chunk = text[j:j+chunk_size_local].strip()
            if chunk:
                local_chunks.append(chunk)
                local_metadata.append((filename, i+1))
    return full_text, local_chunks, local_metadata

def extract_file_id_from_link(link):
    # Try to extract file id from common Drive share formats
    import re
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",  # /d/<id>/
        r"id=([a-zA-Z0-9_-]+)",  # ?id=<id>
        r"folders/([a-zA-Z0-9_-]+)"  # folders/<id>
    ]
    for p in patterns:
        m = re.search(p, link)
        if m:
            return m.group(1)
    return None

# --------------------------
# Processing logic (batched)
# --------------------------
process_button = st.button("Start processing selected PDFs")
if process_button:
    chunks = []
    metadata = []
    progress = st.progress(0)
    total_files = 0
    file_entries = []

    # gather files from folder
    if folder_id:
        folder_files = list_pdfs_in_folder(folder_id)
        file_entries.extend(folder_files)
    # drive link
    if drive_link:
        fid = extract_file_id_from_link(drive_link.strip())
        if fid:
            file_entries.append({"id": fid, "name": f"drive_file_{fid}.pdf"})
    # uploaded files
    if uploaded_files:
        for f in uploaded_files:
            file_entries.append({"upload_obj": f, "name": f.name})

    total_files = len(file_entries)
    if total_files == 0:
        st.warning("No PDF files found. Provide a folder ID, a file link, or upload files.")
    else:
        for idx, fe in enumerate(file_entries):
            if "id" in fe:
                pdf_bytes = download_pdf(fe["id"])
                if pdf_bytes is None:
                    continue
                filename = fe.get("name", fe["id"])
            else:
                # uploaded file object
                pdf_bytes = fe["upload_obj"].read()
                filename = fe.get("name", "uploaded.pdf")
            st.info(f"Processing {filename} ({idx+1}/{total_files})")
            _, local_chunks, local_metadata = extract_text_from_pdf_bytes(
                pdf_bytes, filename, page_range=pages_input, chunk_size_local=chunk_size
            )
            chunks.extend(local_chunks)
            metadata.extend(local_metadata)
            progress.progress(int((idx+1)/total_files * 100))

        st.success(f"Processing finished. Total chunks: {len(chunks)}")

        # Build embeddings and FAISS index
        if len(chunks) > 0:
            with st.spinner("Building embeddings (this may take a minute)..."):
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode(chunks, show_progress_bar=False)
                d = embeddings.shape[1]
                index = faiss.IndexFlatL2(d)
                index.add(embeddings)
            st.session_state["chunks"] = chunks
            st.session_state["metadata"] = metadata
            st.session_state["index"] = index
            st.session_state["model"] = model
        else:
            st.warning("No text extracted from the provided PDFs.")

# --------------------------
# Query interface (chat-like)
# --------------------------
if "index" in st.session_state and "model" in st.session_state:
    st.subheader("Ask questions (semantic search)")
    user_query = st.text_input("Enter your question (e.g., 'Inspector name'):")
    if user_query:
        with st.spinner("Searching..."):
            model = st.session_state["model"]
            index = st.session_state["index"]
            qv = model.encode([user_query])
            D, I = index.search(qv, k=int(top_k))
            results = []
            for i in I[0]:
                file_name, page_num = st.session_state["metadata"][i]
                snippet = st.session_state["chunks"][i]
                results.append({"File": file_name, "Page": page_num, "Text": snippet})
            # Show results
            for r in results:
                st.write(f"**{r['File']} (page {r['Page']})**")
                st.write(r["Text"])
                st.markdown("---")

    # Export results (all chunks) to Excel
    if st.button("Export all extracted text to Excel"):
        df = pd.DataFrame({
            "File Name": [m[0] for m in st.session_state["metadata"]],
            "Page": [m[1] for m in st.session_state["metadata"]],
            "Text": st.session_state["chunks"]
        })
        excel_path = "OCR_results.xlsx"
        df.to_excel(excel_path, index=False)
        st.download_button("Download Excel", open(excel_path, "rb"), file_name="OCR_results.xlsx")
