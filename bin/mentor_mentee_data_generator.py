# Here given a pdf from mentor mentee dataset, we convert it into text formatting (html if possible since GPT has better performance with html). Given mentor VIICTOR profile, we ask GPT to generate a mock CV for a student who might be interested in working with given mentor.
import openai
import pandas as pd
from openai import OpenAI
import PyPDF2
import os
import re
#from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from transformers import GPT2TokenizerFast
#load_dotenv()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
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

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    return text
    
    
def generate_samples(prompt):
    #llm = OpenAI(openai_api_key=api_key)
    chain = load_qa_chain(client,verbose=True)
    question = prompt
    response = chain.run(question=prompt)
    return response


folder_path = os.getenv("PDF_FILE_PATH")

# List to store the data
data = []

# Iterate over all PDFs in the folder
# Iterate over all PDFs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        #Load the PDF
        loader = PyPDFLoader(pdf_path)
        max_tokens = 3500
        mentor_profile_documents = loader.load()

        # Extract text content from Document objects
        mentor_profile_text_list = [doc.page_content for doc in mentor_profile_documents]

        # Join the list into a single string
        mentor_profile_text = " ".join(mentor_profile_text_list)

        mentor_profile_text = preprocess_text(mentor_profile_text)
        mentor_profile_text = truncate_text(mentor_profile_text, max_tokens)


        #mentor_profile_text = pdf_to_text(pdf_path)

        # Preprocess mentor profile text
        # mentor_profile_text = preprocess_text(mentor_profile_text)

        # Truncate mentor profile if too long
        # max_length = 1000  # Adjust as needed to fit within the model's context limit
        # if len(mentor_profile_text) > max_length:
        #    mentor_profile_text = mentor_profile_text[:max_length]

        # print("Mentor Profile Text:", mentor_profile_text)  # Debugging print statement

        # Create prompt for generating a mock CV
        prompt_mentor_mentee = f"""
        Generate a mock CV for a student who might be interested in working with this mentor. The CV should highlight relevant skills, education, and experience that align with the mentor's expertise.
        Only answer with CV, DO NOT inlcude any notes or additional text after CV. Generate mock personal information and school names. DO NOT state the name of the mentor in any part of the CV.
        
        Given the following mentor profile:
        {mentor_profile_text}

        """

        # Generate mock CV samples
        #n_samples = 1  # Number of samples to generate
        mock_cvs = generate_samples(prompt_mentor_mentee)

        print("Generated CVs:", mock_cvs)  # Debugging print statement

        # Append the mentor profile and generated CV to the data list
        for cv in mock_cvs:
            data.append({"Mentor Profile": mentor_profile_text, "Mock Student CV": cv})

# Check if data list is empty
if not data:
    print("No data to save. Exiting...")
else:
    # Create a DataFrame and save to a .csv file
    df = pd.DataFrame(data)
    df.to_csv("mentor_student_cvs_final.csv", index=False)  # Change to .tsv if needed

    print("CSV file has been created successfully.")
