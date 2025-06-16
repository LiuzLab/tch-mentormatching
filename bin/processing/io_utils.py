import os
import glob
import csv
import json
from PyPDF2 import PdfReader
from docx import Document
from preprocessing.text_utils import truncate_text


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_documents(directory: str, extensions: list[str] = None) -> list[tuple[str, str]]:
    """
    Load all documents in the directory matching given extensions.
    Returns list of (filename, text).
    """
    if extensions is None:
        extensions = [".pdf", ".docx", ".txt"]
    docs: list[tuple[str, str]] = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        for path in glob.glob(pattern):
            if ext == ".pdf":
                text = extract_text_from_pdf(path)
            elif ext == ".docx":
                text = extract_text_from_docx(path)
            elif ext == ".txt":
                text = extract_text_from_txt(path)
            else:
                continue
            docs.append((os.path.basename(path), text))
    return docs


def convert_txt_dir_to_csv(input_pattern: str, output_csv: str) -> None:
    """
    Read all txt files matching glob input_pattern and write to CSV with columns [filename, content].
    """
    txt_files = glob.glob(input_pattern)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "content"])
        for path in txt_files:
            try:
                content = extract_text_from_txt(path)
                writer.writerow([os.path.basename(path), content])
            except Exception as e:
                print(f"Error processing {path}: {e}")


def prepare_batches(texts: list[str], batch_size: int = 5, max_tokens: int = 3000) -> list[list[str]]:
    """
    Chunk and truncate texts into batches for API calls.
    Returns list of text batches.
    """
    batches: list[list[str]] = []
    for i in range(0, len(texts), batch_size):
        batch = [truncate_text(t, max_tokens) for t in texts[i : i + batch_size]]
        batches.append(batch)
    return batches


def save_json(data, output_path: str) -> None:
    """
    Save arbitrary data (e.g., summaries) to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

