import os
import glob
import csv
import json
from PyPDF2 import PdfReader
from docx import Document
from .text_utils import truncate_text, clean_and_validate_text


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_documents(
    directory: str, extensions: list[str] = None
) -> list[tuple[str, str]]:
    """
    Load all documents in the directory and its subdirectories matching given extensions.
    Returns list of (filename, text).
    """
    if extensions is None:
        extensions = [".pdf", ".docx", ".txt"]
    docs: list[tuple[str, str]] = []
    for ext in extensions:
        pattern = os.path.join(directory, f"**/*{ext}")
        for path in glob.glob(pattern, recursive=True):
            raw_text = ""
            try:
                if ext == ".pdf":
                    raw_text = extract_text_from_pdf(path)
                elif ext == ".docx":
                    raw_text = extract_text_from_docx(path)
                elif ext == ".txt":
                    raw_text = extract_text_from_txt(path)

                cleaned_text = clean_and_validate_text(raw_text)

                if not cleaned_text:
                    print(
                        f"Warning: Could not extract usable text from {os.path.basename(path)}. Skipping file."
                    )
                    continue
            except Exception as e:
                print(
                    f"Error processing file {os.path.basename(path)}: {e}. Skipping file."
                )
                continue

            docs.append((os.path.basename(path), cleaned_text))
    return docs


def load_document(file_path: str) -> str | None:
    """
    Load a single document from the given file path.
    Returns the text content of the document.
    """
    ext = os.path.splitext(file_path)[1]
    raw_text = None
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        raw_text = extract_text_from_docx(file_path)
    elif ext == ".txt":
        raw_text = extract_text_from_txt(file_path)

    if raw_text:
        return clean_and_validate_text(raw_text)
    return None


def prepare_batches(
    texts: list[str], batch_size: int = 5, max_tokens: int = 3000
) -> list[list[str]]:
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
