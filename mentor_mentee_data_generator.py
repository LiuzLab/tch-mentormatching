# Here given a pdf from mentor mentee dataset, we convert it into text formatting (html if possible since GPT has better performance with html). Given mentor VIICTOR profile, we ask GPT to generate a mock CV for a student who might be interested in working with given mentor. 
import openai
import pandas as pd
from openai import OpenAI
import PyPDF2
import os
import re
# Set your OpenAI API key here
#openai.api_key = 'insert your key here'
client = OpenAI(
    # This is the default and can be omitted
    api_key="insert your key here")



def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


def preprocess_text(text):
    # Replace tabs with a single space
    text = text.replace('\t', ' ')
    
    # Remove newlines
    text = text.replace('\n', ' ')
    
    # Remove multiple spaces and strip leading/trailing spaces
    text = re.sub(' +', ' ', text).strip()
    
    return text
    
    
def generate_samples(prompt, n_samples):
    samples = []

    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=1.0
        )
        samples.append(response.choices[0].message.content)
    return samples


# Folder containing the PDFs
folder_path = '/mnt/belinda_local/daniel/home/MentorMenteeSimulateddata/final'

# List to store the data
data = []

# Iterate over all PDFs in the folder
# Iterate over all PDFs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, filename)
        mentor_profile_text = pdf_to_text(pdf_path)
        
        # Preprocess mentor profile text
        mentor_profile_text = preprocess_text(mentor_profile_text)
        
        # Truncate mentor profile if too long
        max_length = 1000  # Adjust as needed to fit within the model's context limit
        if len(mentor_profile_text) > max_length:
            mentor_profile_text = mentor_profile_text[:max_length]
        
        #print("Mentor Profile Text:", mentor_profile_text)  # Debugging print statement
        
        # Create prompt for generating a mock CV
        prompt_mentor_mentee = f"""
        Generate a mock CV for a student who might be interested in working with this mentor. The CV should highlight relevant skills, education, and experience that align with the mentor's expertise.
        Only answer with CV, DO NOT inlcude any notes or additional text after CV. Generate mock personal information and school names. DO NOT state the name of the mentor in any part of the CV.
        
        Given the following mentor profile:

        {mentor_profile_text}

        """
        
        # Generate mock CV samples
        n_samples = 1  # Number of samples to generate
        mock_cvs = generate_samples(prompt_mentor_mentee, n_samples)
        
        print("Generated CVs:", mock_cvs)  # Debugging print statement
        
        # Append the mentor profile and generated CV to the data list
        for cv in mock_cvs:
            data.append({
                'Mentor Profile': mentor_profile_text,
                'Mock Student CV': cv
            })

# Check if data list is empty
if not data:
    print("No data to save. Exiting...")
else:
    # Create a DataFrame and save to a .csv file
    df = pd.DataFrame(data)
    df.to_csv('mentor_student_cvs_final.csv', index=False)  # Change to .tsv if needed

    print("CSV file has been created successfully.")
