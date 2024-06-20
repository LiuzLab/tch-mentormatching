import openai
import pandas as pd
from openai import OpenAI
import PyPDF2
import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# API key for OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def pdf_to_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def preprocess_text(text):
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re.sub(" +", " ", text).strip()
    return text


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def generate_samples(prompt, documents):
    chain = load_qa_chain(client, verbose=True)
    responses = []
    for doc in documents:
        response = chain.run(input_documents=[doc], question=prompt)
        responses.append(response)
    return responses


# Folder containing the PDFs
folder_path = os.getenv("PDF_FILE_PATH")


# List to store the data
data = []

# Iterate over all PDFs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(pdf_path)
        mentor_profile_text = loader.load()

        # Preprocess mentor profile text
        mentor_profile_text = preprocess_text(mentor_profile_text)

        # Chunk the mentor profile text
        chunks = chunk_text(mentor_profile_text, chunk_size=1000, chunk_overlap=200)

        # Create prompt for generating a mock CV
        prompt_mentor_mentee = f"""
        Generate a mock CV for a student who might be interested in working with this mentor. The CV should highlight relevant skills, education, and experience that align with the mentor's expertise.
        Only answer with CV, DO NOT include any notes or additional text after CV. Generate mock personal information and school names. DO NOT state the name of the mentor in any part of the CV.

        Given the following mentor profile:
        """

        # Generate mock CV samples
        mock_cvs = generate_samples(prompt_mentor_mentee, chunks)

        # Aggregate the responses from each chunk
        aggregated_cv = " ".join(mock_cvs)

        # Append the mentor profile and generated CV to the data list
        data.append(
            {"Mentor Profile": mentor_profile_text, "Mock Student CV": aggregated_cv}
        )

# Check if data list is empty
if not data:
    print("No data to save. Exiting...")
else:
    # Create a DataFrame and save to a .csv file
    df = pd.DataFrame(data)
    df.to_csv("mentor_student_cvs_final.csv", index=False)

    print("CSV file has been created successfully.")
