import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
import io
import requests
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# =========================
# Load Secrets
# =========================
SERVICE_INFO = st.secrets["SERVICE_ACCOUNT_JSON"]
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]["token"]

# Google Drive credentials
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
credentials = service_account.Credentials.from_service_account_info(
    SERVICE_INFO, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)

# Hugging Face API
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# =========================
# Settings
# =========================
PDF_DPI = 150
MAX_CHARS_PER_FILE = 1000  # for AI queries

# =========================
# Helper: Parse page ranges
# =========================
def parse_page_selection(page_input, total_pages):
    """
    Convert a user string like "1,2,5-7" into a list of integers.
    Ignores invalid numbers or numbers > total_pages.
    """
    pages = set()
    parts = page_input.split(',')
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            try:
                start = int(start)
                end = int(end)
                pages.update(range(start, end + 1))
            except:
                continue
        else:
            try:
                pages.add(int(part))
            except:
                continue
    return sorted([p for p in pages if 1 <= p <= total_pages])

# =========================
# OCR Extraction
# =========================
def extract_text_from_pdf_bytes(pdf_bytes, pages=None):
    images = convert_from_bytes(pdf_bytes, dpi=PDF_DPI)
    if pages:
        images = [img for i, img in enumerate(images, start=1) if i in pages]
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="eng") + "\n"
    return text

# =========================
# Hugging Face Chat
# =========================
def query_hf(prompt, context):
    payload = {
        "inputs": f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:",
        "parameters": {"max_new_tokens": 300, "temperature": 0.3}
    }
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"[Error from Hugging Face] {str(e)}"

# =========================
# Extract File or Folder ID
# =========================
def extract_drive_id(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1), "file"
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1), "file"
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1), "folder"
    return None, None

# =========================
# List PDFs in a Google Drive Folder
# =========================
def list_pdfs_in_folder(folder_id):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id, name)"
    ).execute()
    return results.get('files', [])

# =========================
# Download PDF Bytes in Memory
# =========================
def download_pdf_bytes(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    file_data.seek(0)
    return file_data.getvalue()

# =========================
# Streamlit UI
# =========================
st.title("📄 Nesma PDF OCR + AI Chat")

choice = st.radio("Select input:", ["📂 Upload PDFs", "🔗 Google Drive Link"])
all_texts = []  # store (file_name, text) for all PDFs

# -------------------------
# Upload PDFs
# -------------------------
if choice == "📂 Upload PDFs":
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded_files:
        for pdf in uploaded_files:
            pdf_bytes = pdf.read()
            total_pages = len(convert_from_bytes(pdf_bytes, dpi=PDF_DPI))
            page_input = st.text_input(
                f"Enter pages for {pdf.name} (e.g., 1,3-5, default all):", "all"
            )
            if page_input.strip().lower() == "all":
                pages_selected = list(range(1, total_pages + 1))
            else:
                pages_selected = parse_page_selection(page_input, total_pages)
            if pages_selected:
                with st.spinner(f"Extracting text from {pdf.name}..."):
                    text = extract_text_from_pdf_bytes(pdf_bytes, pages_selected)
                all_texts.append((pdf.name, text))

# -------------------------
# Google Drive Link or Folder
# -------------------------
elif choice == "🔗 Google Drive Link":
    drive_link = st.text_input("Paste Google Drive file or folder link:")
    if drive_link:
        drive_id, link_type = extract_drive_id(drive_link)
        if not drive_id:
            st.error("Invalid Google Drive link. Please use a proper file or folder link.")
        else:
            files_to_process = []
            if link_type == "file":
                files_to_process = [{"id": drive_id, "name": "drive_file.pdf"}]
            elif link_type == "folder":
                files_to_process = list_pdfs_in_folder(drive_id)
                if not files_to_process:
                    st.warning("No PDF files found in this folder.")
            for f in files_to_process:
                pdf_bytes = download_pdf_bytes(f['id'])
                total_pages = len(convert_from_bytes(pdf_bytes, dpi=PDF_DPI))
                page_input = st.text_input(
                    f"Enter pages for {f['name']} (e.g., 1,3-5, default all):", "all"
                )
                if page_input.strip().lower() == "all":
                    pages_selected = list(range(1, total_pages + 1))
                else:
                    pages_selected = parse_page_selection(page_input, total_pages)
                if pages_selected:
                    with st.spinner(f"Extracting text from {f['name']}..."):
                        text = extract_text_from_pdf_bytes(pdf_bytes, pages_selected)
                    all_texts.append((f['name'], text))

# -------------------------
# Display Text and AI Chat
# -------------------------
if all_texts:
    df = pd.DataFrame(all_texts, columns=["File", "Content"])
    st.subheader("Extracted Text")
    st.dataframe(df)

    question = st.text_input("💬 Ask AI about these PDFs:")
    if st.button("Ask AI") and question:
        # Limit context per file
        context = "\n\n".join([t[1][:MAX_CHARS_PER_FILE] for t in all_texts])
        with st.spinner("Querying AI..."):
            answer = query_hf(question, context)
        st.write("### 🤖 AI Answer:")
        st.write(answer)

    if st.button("📥 Export to Excel"):
        out_path = "results.xlsx"
        df.to_excel(out_path, index=False)
        with open(out_path, "rb") as f:
            st.download_button("Download Excel", f, file_name="results.xlsx")