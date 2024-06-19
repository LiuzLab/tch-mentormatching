import os
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY')

# Initialize OpenAI client
openai.api_key = api_key

# Read the CSV file
file_path = './data/mentor_student_cvs_final.csv'
data = pd.read_csv(file_path, usecols=[0, 1])

# Retry decorator to handle API rate limits
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Function to process CVs
def process_cvs(data):
    results = []
    instructions = "Based on the following text, generate a summary paragraph. While constructing the paragraph, carefully extract key entities, interests of the individual, and anything else you deem important about that individual, from the text to include in your summary."
    system_prompt = {"role": "system", "content": "You are a helpful assistant skilled at summarizing and extracting key information from text."}

    for _, row in data.iterrows():
        mentor_cv = f"{instructions}\n{row[0]}"
        student_cv = f"{instructions}\n{row[1]}"

        mentor_message = [system_prompt, {"role": "user", "content": mentor_cv}]
        student_message = [system_prompt, {"role": "user", "content": student_cv}]

        # Get completions with backoff
        mentor_completion = completion_with_backoff(model="gpt-4", messages=mentor_message, temperature=1.0)
        student_completion = completion_with_backoff(model="gpt-4", messages=student_message, temperature=1.0)

        mentor_completion_content = mentor_completion['choices'][0]['message']['content']
        student_completion_content = student_completion['choices'][0]['message']['content']

        results.append([mentor_completion_content, student_completion_content])

    result_df = pd.DataFrame(results, columns=['Mentor CV Completion', 'Student CV Completion'])
    return result_df

# Process the CVs and get the new dataframe
result_df = process_cvs(data)

# Save the results to a new CSV file
result_df.to_csv('./data/mentor_student_research_summaries.csv', index=False)
