import openai
import pandas as pd
from openai import OpenAI
import os
import random
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
import asyncio
from docx import Document
from langchain.docstore.document import Document as LangchainDocument

load_dotenv()

client = ChatOpenAI(
    model="gpt-4",
    temperature=1.25,
    max_tokens=3000,
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def generate_samples(prompt, mentor_profile_documents):
    chain = load_qa_chain(client, verbose=True)
    response = await chain.arun(input_documents=mentor_profile_documents, question=prompt)
    return response

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i in range(min(2, len(reader.pages))):
        page = reader.pages[i]
        text += page.extract_text()
    text = " ".join(text.split())
    return text

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def load_docx(file_path):
    text = extract_text_from_docx(file_path)
    return [LangchainDocument(page_content=text, metadata={"source": file_path})]


async def generate_mock_cv(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.docx':
        file_text = extract_text_from_docx(file_path)
        mentor_profile_documents = load_docx(file_path)
    elif file_extension == '.pdf':
        file_text = extract_text_from_pdf(file_path)
        mentor_profile_documents = load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Please use .docx or .pdf")

    n_papers = random.randint(1, 3)
    n_experiences = random.randint(1, 3)
    n_skills = random.randint(2, 5)
    n_volunteering = random.randint(0, 2)
    education = random.choice(["BS and Master's Degree", "Only BS Degree"])

    prompt_mentor_mentee = f"""
    Generate a mock CV for a student who might be interested in working with this mentor. The CV should highlight relevant skills, education, and experience that align with the mentor's expertise.
    Only answer with CV, DO NOT include any notes or additional text after CV. Generate mock personal information and school names. DO NOT state the name of the mentor in any part of the CV.

    Consider a CV with only {n_papers} number of papers, {n_experiences} number of experiences, {n_skills} number of skills, and {n_volunteering} number of volunteering activities.
    Consider the education level as: {education}.

    Given the following mentor profile:
    """

    chain = load_qa_chain(client, verbose=True)
    mock_cv = await chain.arun(input_documents=mentor_profile_documents[0:1], question=prompt_mentor_mentee)
    
    return mock_cv, file_text



if __name__ == "__main__":
    # Add any test code here if needed
    pass
