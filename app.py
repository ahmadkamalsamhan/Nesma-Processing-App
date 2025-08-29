import streamlit as st
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
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
# OCR Extraction
# =========================
def extract_text_from_pdf_bytes(pdf_bytes, filename, page_range=None):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    text = ""
    for i, img in enumerate(images, start=1):
        if page_range and i not in page_range:
            continue
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
# Google Drive Downloader
# =========================
def download_drive_file(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    file_data.seek(0)
    return file_data.getvalue()

# =========================
# Extract File or Folder ID
# =========================
def extract_drive_id(url):
    """
    Returns a tuple: (id, type) where type is 'file' or 'folder'
    Supports:
    - file links: /d/FILE_ID or open?id=FILE_ID
    - folder links: /folders/FOLDER_ID
    """
    # Check file /d/ or open?id=
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1), "file"
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1), "file"
    # Check folder /folders/
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
    files = results.get('files', [])
    return files

# =========================
# Streamlit UI
# =========================
st.title("📄 Nesma PDF OCR + AI Chat")

choice = st.radio("Select input:", ["📂 Upload PDFs", "🔗 Google Drive Link"])
all_texts = []

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
            with st.spinner(f"Processing {pdf.name}..."):
                text = extract_text_from_pdf_bytes(pdf_bytes, pdf.name)
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
            if link_type == "file":
                try:
                    with st.spinner("Downloading file from Google Drive..."):
                        pdf_bytes = download_drive_file(drive_id)
                    text = extract_text_from_pdf_bytes(pdf_bytes, "drive_file.pdf")
                    all_texts.append(("drive_file.pdf", text))
                except Exception as e:
                    st.error(f"Google Drive Error: {str(e)}")
            elif link_type == "folder":
                try:
                    with st.spinner("Listing PDFs in folder..."):
                        files = list_pdfs_in_folder(drive_id)
                    if not files:
                        st.warning("No PDF files found in this folder.")
                    for f in files:
                        with st.spinner(f"Downloading {f['name']}..."):
                            pdf_bytes = download_drive_file(f['id'])
                            text = extract_text_from_pdf_bytes(pdf_bytes, f['name'])
                            all_texts.append((f['name'], text))
                except Exception as e:
                    st.error(f"Google Drive Folder Error: {str(e)}")

# -------------------------
# Display Text and AI Chat
# -------------------------
if all_texts:
    df = pd.DataFrame(all_texts, columns=["File", "Content"])
    st.subheader("Extracted Text")
    st.dataframe(df)

    question = st.text_input("💬 Ask AI about these PDFs:")
    if st.button("Ask AI") and question:
        context = "\n\n".join([t[1] for t in all_texts])[:4000]  # keep context short
        with st.spinner("Querying AI..."):
            answer = query_hf(question, context)
        st.write("### 🤖 AI Answer:")
        st.write(answer)

    if st.button("📥 Export to Excel"):
        out_path = "results.xlsx"
        df.to_excel(out_path, index=False)
        with open(out_path, "rb") as f:
            st.download_button("Download Excel", f, file_name="results.xlsx")