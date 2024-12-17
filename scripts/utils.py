import os
import fitz  # PyMuPDF
from docx import Document

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def read_files_from_folder(folder_path):
    files_content = {}
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt') or filename.endswith('.docx') or filename.endswith('.pdf'):
                file_path = os.path.join(root, filename)
                if filename.endswith('.txt'):
                    text = read_txt(file_path)
                elif filename.endswith('.docx'):
                    text = read_docx(file_path)
                elif filename.endswith('.pdf'):
                    text = read_pdf(file_path)
                files_content[file_path] = text
    return files_content
