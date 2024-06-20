# Here given a pdf from mentor mentee dataset, we convert it into text formatting (html if possible since GPT has better performance with html). Given mentor VIICTOR profile, we ask GPT to generate a mock CV for a student who might be interested in working with given mentor.
import openai
import pandas as pd
from openai import OpenAI
import os
import re
import random
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from transformers import GPT2TokenizerFast
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path = "/mnt/belinda_local/daniel/home/github_mentor_mentee_main/.env")

client = ChatOpenAI(
    model="gpt-4o",
    temperature=1.25,
    max_tokens=3000,
    api_key=os.getenv("OPENAI_KEY")) # Add randomness by considering 1.25 temperature instead of 1 previously

def generate_samples(prompt, mentor_profile_documents):
    # llm = OpenAI(openai_api_key=api_key)
    chain = load_qa_chain(client, verbose=True)
    question = prompt
    response = chain.run(input_documents=mentor_profile_documents, question=prompt)
    return response

# Function to extract and preprocess text from the first two pages of a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i in range(min(2, len(reader.pages))):
        page = reader.pages[i]
        text += page.extract_text()
    text = " ".join(text.split())  # Remove extra whitespaces
    return text

# Folder containing the PDFs
folder_path = os.getenv("PDF_FILE_PATH")

# List to store the data
data = []

# Iterate over all PDFs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        mentor_profile_documents = loader.load_and_split()

        # Generate random numbers for papers, experiences, skills, and volunteering
        n_papers = random.randint(1, 3)
        n_experiences = random.randint(1, 3)
        n_skills = random.randint(2, 5)
        n_volunteering = random.randint(0, 2)

        # Randomly select education level
        education_levels = ["BS and Master's Degree", "Only BS Degree"]
        education = random.choice(education_levels)

        prompt_mentor_mentee = f"""
        Generate a mock CV for a student who might be interested in working with this mentor. The CV should highlight relevant skills, education, and experience that align with the mentor's expertise.
        Only answer with CV, DO NOT include any notes or additional text after CV. Generate mock personal information and school names. DO NOT state the name of the mentor in any part of the CV.

        Consider a CV with only {n_papers} number of papers, {n_experiences} number of experiences, {n_skills} number of skills, and {n_volunteering} number of volunteering activities.
        Consider the education level as: {education}.

        Given the following mentor profile:
        """

        # Generate mock CV samples
        mock_cv = generate_samples(prompt_mentor_mentee, mentor_profile_documents[0:1])

        data.append({"Mentor Profile": filename, "Mock Student CV": mock_cv, "PDF Text": pdf_text})

# Check if data list is empty
if not data:
    print("No data to save. Exiting...")
else:
    # Create a DataFrame and save to a .csv file
    df = pd.DataFrame(data)
    df.to_csv("../simulated_data/mentor_student_cvs_final.csv", index=False)  # Change to .tsv if needed

    print("CSV file has been created successfully.")
