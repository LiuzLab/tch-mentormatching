# Here given a pdf from mentor mentee dataset, we convert it into text formatting (html if possible since GPT has better performance with html). Given mentor VIICTOR profile, we ask GPT to generate a mock CV for a student who might be interested in working with given mentor.
import openai
import pandas as pd
from openai import OpenAI
import PyPDF2
import os
import re

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from transformers import GPT2TokenizerFast

load_dotenv(dotenv_path = "")

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def generate_samples(prompt, mentor_profile_documents):
    # llm = OpenAI(openai_api_key=api_key)
    chain = load_qa_chain(client, verbose=True)
    question = prompt
    response = chain.run(input_documents=mentor_profile_documents, question=prompt)
    return response


# Folder containing the PDFs
folder_path = os.getenv("PDF_FILE_PATH")

# List to store the data
data = []

# Iterate over all PDFs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        mentor_profile_documents = loader.load_and_split()

        prompt_mentor_mentee = f"""
        Generate a mock CV for a student who might be interested in working with this mentor. The CV should highlight relevant skills, education, and experience that align with the mentor's expertise.
        Only answer with CV, DO NOT inlcude any notes or additional text after CV. Generate mock personal information and school names. DO NOT state the name of the mentor in any part of the CV.

        Given the following mentor profile:

        """

        # Generate mock CV samples
        mock_cv = generate_samples(prompt_mentor_mentee, mentor_profile_documents[0:1])

        data.append({"Mentor Profile": filename, "Mock Student CV": mock_cv})

# Check if data list is empty
if not data:
    print("No data to save. Exiting...")
else:
    # Create a DataFrame and save to a .csv file
    df = pd.DataFrame(data)
    df.to_csv("mentor_student_cvs_final.csv", index=False)  # Change to .tsv if needed

    print("CSV file has been created successfully.")
