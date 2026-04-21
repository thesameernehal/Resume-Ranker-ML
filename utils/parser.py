import fitz  
import docx
import os


def read_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text


def read_docx(file_path):
    text = ""
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.lower()
    return text


def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = read_pdf(file_path)
    elif ext == ".docx":
        text = read_docx(file_path)
    elif ext == ".txt":
        text = read_txt(file_path)
    else:
        return ""

    return clean_text(text)

extract_text = extract_text_from_file