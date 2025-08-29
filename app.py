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
import json
import re

# tell pytesseract where the tesseract binary is (Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Nesma PDF OCR + AI Chat", layout="wide")
st.title("📄 Nesma PDF OCR + Free AI Chat (RAG)")

# ----------------- Secrets: service account + HF token -----------------
if "SERVICE_ACCOUNT_JSON" not in st.secrets:
    st.error("SERVICE_ACCOUNT_JSON not found in Streamlit secrets. Add it in Manage App → Secrets.")
    st.stop()

# service account secret can be a dict (TOML) or a JSON string
sa_secret = st.secrets["SERVICE_ACCOUNT_JSON"]
if isinstance(sa_secret, str):
    try:
        service_account_info = json.loads(sa_secret)
    except Exception:
        st.error("SERVICE_ACCOUNT_JSON appears to be a string but not valid JSON. Use TOML section or valid JSON string.")
        st.stop()
elif isinstance(sa_secret, dict):
    service_account_info = dict(sa_secret)
else:
    st.error("Unsupported SERVICE_ACCOUNT_JSON format in secrets.")
    st.stop()

# Repair private_key formatting if it contains escaped \n:
if "private_key" in service_account_info:
    pk = service_account_info["private_key"]
    if isinstance(pk, str) and "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    if isinstance(pk, str) and not pk.endswith("\n"):
        pk += "\n"
    service_account_info["private_key"] = pk

# Build credentials
SCOPES = ["https://www.googleapis.com/auth/drive"]
try:
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)
except Exception:
    st.error("Failed to initialize Google Drive service with service account. Check secrets formatting and that service account has folder access.")
    st.text(traceback.format_exc())
    st.stop()

# Hugging Face token (optional — if missing we fallback to local summarization)
HF_TOKEN = st.secrets.get("HF_API_TOKEN", None)
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # sample free model; you can change

# ----------------- UI inputs -----------------
st.markdown("**Input PDFs** — paste a folder ID, paste a single file link, or upload PDFs.")
col1, col2 = st.columns(2)
with col1:
    folder_id = st.text_input("Google Drive Folder ID (optional)")
    drive_link = st.text_input("Google Drive file link (optional)")
with col2:
    uploaded_files = st.file_uploader("Upload PDFs (optional)", type="pdf", accept_multiple_files=True)

pages_input = st.text_input("Page range for processing (e.g., 'all' or '1-5')", value="all")
chunk_size = st.number_input("Chunk size (characters)", value=500, min_value=200, max_value=2000, step=100)
top_k = st.number_input("Top K retrieval (for RAG)", value=5, min_value=1, max_value=20, step=1)
process_button = st.button("🚀 Start processing (OCR + Embeddings)")

# ----------------- Helper functions -----------------
def extract_file_id_from_link(link):
    patterns = [r"/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"folders/([a-zA-Z0-9_-]+)"]
    for p in patterns:
        m = re.search(p, link)
        if m:
            return m.group(1)
    return None

def list_pdfs_in_folder(fid):
    try:
        files = []
        page_token = None
        while True:
            resp = drive_service.files().list(
                q=f"'{fid}' in parents and mimeType='application/pdf'",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                pageSize=1000
            ).execute()
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return files
    except Exception:
        st.error("Failed to list files in folder. Ensure folder ID is correct and service account has access.")
        st.text(traceback.format_exc())
        return []

def download_pdf_bytes(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.content
    else:
        st.error(f"Failed to download file {file_id} (HTTP {r.status_code}). Ensure file is shared with the service account or publicly).")
        return None

def extract_text_from_pdf_bytes(file_bytes, filename, page_range="all", chunk_size_local=500):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        st.error(f"Cannot open PDF {filename}. It may be corrupted.")
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
    local_meta = []
    for i in pages:
        page = doc[i]
        pix = page.get_pixmap(dpi=200)  # DPI tradeoff speed vs accuracy
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            text = pytesseract.image_to_string(img, lang="ara+eng")
        except Exception:
            text = pytesseract.image_to_string(img)
        full_text += text + "\n"
        for j in range(0, len(text), chunk_size_local):
            chunk = text[j:j+chunk_size_local].strip()
            if chunk:
                local_chunks.append(chunk)
                local_meta.append((filename, i+1))
    return full_text, local_chunks, local_meta

# ----------------- Processing (OCR -> chunks -> embeddings) -----------------
if process_button:
    entries = []
    if folder_id:
        folder_files = list_pdfs_in_folder(folder_id)
        entries.extend(folder_files)
    if drive_link:
        fid = extract_file_id_from_link(drive_link.strip())
        if fid:
            entries.append({"id": fid, "name": f"{fid}.pdf"})
    if uploaded_files:
        for f in uploaded_files:
            entries.append({"upload_obj": f, "name": f.name})

    if not entries:
        st.warning("No PDFs provided. Please provide a folder ID, file link, or upload files.")
    else:
        chunks = []
        metadata = []
        progress = st.progress(0)
        total = len(entries)
        for idx, e in enumerate(entries):
            if "id" in e:
                b = download_pdf_bytes(e["id"])
                name = e.get("name", e["id"])
            else:
                b = e["upload_obj"].read()
                name = e["name"]
            if not b:
                continue
            _, local_chunks, local_meta = extract_text_from_pdf_bytes(b, name, page_range=pages_input, chunk_size_local=chunk_size)
            chunks.extend(local_chunks)
            metadata.extend(local_meta)
            progress.progress(int((idx + 1) / total * 100))

        st.success(f"Extracted {len(chunks)} chunks from {total} file(s).")
        # Build embeddings + faiss
        if len(chunks) == 0:
            st.warning("No text was extracted from files.")
        else:
            with st.spinner("Building embeddings (may take a minute)..."):
                st.session_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = st.session_state["model"].encode(chunks, show_progress_bar=False)
                d = embeddings.shape[1]
                index = faiss.IndexFlatL2(d)
                index.add(embeddings)
                st.session_state["chunks"] = chunks
                st.session_state["metadata"] = metadata
                st.session_state["index"] = index
            st.success("Embeddings ready — you can now ask questions in the chat below.")

# ----------------- RAG chat: local or Hugging Face -----------------
def call_hf_model(prompt, max_new_tokens=256):
    if not HF_TOKEN:
        return None, "Hugging Face token not configured in secrets (HF_API_TOKEN)."
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    # Use instruction-tuned model endpoint. You can update HF_MODEL to other models.
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        try:
            out = resp.json()
            # response format may vary: sometimes list with generated_text; handle common cases:
            if isinstance(out, list) and "generated_text" in out[0]:
                return out[0]["generated_text"], None
            if isinstance(out, dict) and "generated_text" in out:
                return out["generated_text"], None
            # fallback to raw text
            return resp.text, None
        except Exception as e:
            return None, f"Failed to parse HF response: {e}"
    else:
        return None, f"Hugging Face API error {resp.status_code}: {resp.text}"

st.markdown("---")
st.subheader("💬 Chat with your PDFs (RAG)")

if "index" in st.session_state and "model" in st.session_state:
    user_q = st.text_input("Ask a question (e.g. 'Who is the inspector?')", key="userq")
    if st.button("Ask"):
        if not user_q:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant text..."):
                qv = st.session_state["model"].encode([user_q])
                D, I = st.session_state["index"].search(qv, k=int(top_k))
                context = ""
                for i in I[0]:
                    fname, pnum = st.session_state["metadata"][i]
                    snippet = st.session_state["chunks"][i]
                    context += f"\nFrom {fname} (page {pnum}):\n{snippet}\n"
            # Build a short prompt
            prompt = (
                "You are an assistant that answers user questions using ONLY the provided context excerpts.\n"
                "If the answer is not present, say you couldn't find it.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {user_q}\n\nAnswer concisely:"
            )
            # Call HF model if token configured
            if HF_TOKEN:
                answer, err = call_hf_model(prompt, max_new_tokens=256)
                if err:
                    st.error(err)
                else:
                    st.markdown("### 🤖 Answer (HF model)")
                    st.write(answer)
            else:
                # Fallback: simple heuristic - show top snippets (not generation)
                st.markdown("### 🔎 Top snippets (no HF token configured)")
                for i in I[0]:
                    fname, pnum = st.session_state["metadata"][i]
                    st.markdown(f"**{fname} (page {pnum})**")
                    st.write(st.session_state["chunks"][i])
                    st.markdown("---")
else:
    st.info("Process PDFs first to build the index, then ask questions here.")

# ----------------- Export option -----------------
if "chunks" in st.session_state:
    if st.button("Export all extracted text to Excel"):
        df = pd.DataFrame({
            "File": [m[0] for m in st.session_state["metadata"]],
            "Page": [m[1] for m in st.session_state["metadata"]],
            "Text": st.session_state["chunks"]
        })
        out = BytesIO()
        df.to_excel(out, index=False)
        st.download_button("Download Excel", out.getvalue(), "ocr_results.xlsx")