import os
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI


# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Read the CSV file
file_path = "../simulated_data/mentor_student_cvs_final.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_csv(file_path)


# Retry decorator to handle API rate limits
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def summarize_text(text):
    instructions = (
        "Based on the following text, generate a summary paragraph that includes the following information about the individual: "
        "1. Name "
        "2. Institution "
        "3. Main research interests and areas of expertise "
        "4. Notable publications or achievements "
        "5. Any other relevant details that highlight their professional profile and contributions. "
        "The summary should be concise and informative, making it easy to understand the individual's primary focus and suitability for collaboration or mentorship. "
        "You do not need to list entire publication names just key terms and ideas to inform the summary."
    )
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant skilled at summarizing and extracting key information from text.",
    }

    user_message = {"role": "user", "content": f"{instructions}\n{text}"}

    messages = [system_prompt, user_message]
    response = completion_with_backoff(
        model="gpt-4", messages=messages, temperature=1.0
    )
    return response.choices[0].message.content


# Function to process mentor texts and append summaries to the DataFrame
def process_mentor_text(data):
    summaries = [
        summarize_text(row[2]) for _, row in data.iterrows()
    ]  # Assuming text to summarize is in the third column (index 2)
    data["Mentor_Summary"] = summaries
    return data


# New function to process mentee texts and append summaries to the DataFrame
def process_mentee_text(data):
    summaries = [
        summarize_text(row[1]) for _, row in data.iterrows()
    ]  # Using the second column (index 1) for mentee data
    data["Mentee_Summary"] = summaries
    return data


# Process the mentor and mentee texts and get the new dataframe with summaries
result_df = process_mentor_text(data)
result_df = process_mentee_text(result_df)

# Define the output directory and ensure it exists
output_dir = "../simulated_data"
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(
    output_dir, "mentor_student_cvs_with_summaries_final.csv"
)

# Save the results to a new CSV file
result_df.to_csv(output_file_path, index=False)

print(f"Summarized CVs saved to {output_file_path}")