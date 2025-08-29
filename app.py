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

# ✅ Tell pytesseract where binary is (on Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Nesma PDF OCR + AI", layout="wide")
st.title("📄 Nesma PDF OCR + AI Semantic Search System")

# --------------------------
# Load Google Drive credentials from secrets
# --------------------------
if "SERVICE_ACCOUNT_JSON" not in st.secrets:
    st.error("Missing SERVICE_ACCOUNT_JSON in Streamlit secrets. Please add it in Manage App → Secrets.")
    st.stop()

service_account_info = dict(st.secrets["SERVICE_ACCOUNT_JSON"])

# Fix private_key formatting
if "private_key" in service_account_info:
    pk = service_account_info["private_key"]
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    if not pk.endswith("\n"):
        pk += "\n"
    service_account_info["private_key"] = pk

SCOPES = ["https://www.googleapis.com/auth/drive"]
try:
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    drive_service = build("drive", "v3", credentials=credentials)
except Exception:
    st.error("❌ Failed to create Google Drive credentials. Check secrets formatting.")
    st.text(traceback.format_exc())
    st.stop()

# --------------------------
# UI: PDF sources
# --------------------------
st.subheader("📂 Provide PDF files")

folder_id = st.text_input("Google Drive Folder ID (optional)")
drive_link = st.text_input("Google Drive File Link (optional)")
uploaded_files = st.file_uploader("Upload PDFs (optional)", type="pdf", accept_multiple_files=True)
pages_input = st.text_input("Page range (e.g., 'all' or '1-5')", value="all")

chunk_size = st.number_input("Chunk size (characters)", value=500, min_value=100, max_value=2000, step=100)
top_k = st.number_input("Top K results", value=5, min_value=1, max_value=20, step=1)

# --------------------------
# Helpers
# --------------------------
def list_pdfs_in_folder(fid):
    try:
        results = drive_service.files().list(
            q=f"'{fid}' in parents and mimeType='application/pdf'",
            fields="files(id, name)", pageSize=1000
        ).execute()
        return results.get("files", [])
    except Exception:
        st.error("Failed to list files in folder. Check permissions.")
        st.text(traceback.format_exc())
        return []

def download_pdf(fid):
    url = f"https://drive.google.com/uc?export=download&id={fid}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.content
    else:
        st.error(f"Failed to download file {fid} (status {r.status_code}).")
        return None

def extract_file_id_from_link(link):
    import re
    patterns = [r"/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"folders/([a-zA-Z0-9_-]+)"]
    for p in patterns:
        m = re.search(p, link)
        if m:
            return m.group(1)
    return None

def extract_text_from_pdf_bytes(file_bytes, filename, page_range="all", chunk_size_local=500):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        st.error(f"❌ Cannot open PDF {filename}.")
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
    chunks, metadata = [], []
    for i in pages:
        page = doc[i]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            text = pytesseract.image_to_string(img, lang="ara+eng")
        except:
            text = pytesseract.image_to_string(img)
        full_text += text + "\n"
        for j in range(0, len(text), chunk_size_local):
            chunk = text[j:j+chunk_size_local].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append((filename, i+1))
    return full_text, chunks, metadata

# --------------------------
# Process PDFs
# --------------------------
if st.button("🚀 Start processing"):
    chunks, metadata = [], []
    entries = []

    if folder_id:
        entries.extend(list_pdfs_in_folder(folder_id))
    if drive_link:
        fid = extract_file_id_from_link(drive_link)
        if fid:
            entries.append({"id": fid, "name": f"{fid}.pdf"})
    if uploaded_files:
        for f in uploaded_files:
            entries.append({"upload_obj": f, "name": f.name})

    if not entries:
        st.warning("⚠️ No PDFs provided.")
    else:
        progress = st.progress(0)
        for idx, e in enumerate(entries):
            if "id" in e:
                pdf_bytes = download_pdf(e["id"])
                filename = e.get("name", "drive_file.pdf")
            else:
                pdf_bytes = e["upload_obj"].read()
                filename = e["name"]
            if not pdf_bytes:
                continue
            _, c, m = extract_text_from_pdf_bytes(pdf_bytes, filename, page_range=pages_input, chunk_size_local=chunk_size)
            chunks.extend(c)
            metadata.extend(m)
            progress.progress((idx+1)/len(entries))

        if chunks:
            st.success(f"✅ Extracted {len(chunks)} chunks of text.")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            st.session_state.update({
                "chunks": chunks,
                "metadata": metadata,
                "index": index,
                "model": model
            })
        else:
            st.warning("No text extracted.")

# --------------------------
# Query interface
# --------------------------
if "index" in st.session_state:
    st.subheader("🔎 Ask a Question (Semantic Search)")
    q = st.text_input("Your query here")
    if q:
        model = st.session_state["model"]
        index = st.session_state["index"]
        qv = model.encode([q])
        D, I = index.search(qv, k=top_k)
        for i in I[0]:
            fname, pnum = st.session_state["metadata"][i]
            snippet = st.session_state["chunks"][i]
            st.markdown(f"**📑 {fname} (Page {pnum})**")
            st.write(snippet)
            st.markdown("---")

    if st.button("📥 Export to Excel"):
        df = pd.DataFrame({
            "File": [m[0] for m in st.session_state["metadata"]],
            "Page": [m[1] for m in st.session_state["metadata"]],
            "Text": st.session_state["chunks"]
        })
        out = BytesIO()
        df.to_excel(out, index=False)
        st.download_button("Download Excel", out.getvalue(), "results.xlsx")
