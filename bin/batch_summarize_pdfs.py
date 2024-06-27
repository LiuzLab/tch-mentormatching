import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json
import time
from .generate_text import generate_text

# Instructions for generating the text
mentor_instructions = (
    "Based on the following text, generate a summary paragraph that includes the following information about the individual: "
    "1. Name "
    "2. Institution "
    "3. Main research interests and areas of expertise "
    "4. Notable publications or achievements "
    "5. Any other relevant details that highlight their professional profile and contributions. "
    "The summary should be concise and informative, making it easy to understand the individual's primary focus and suitability for collaboration or mentorship. "
    "You do not need to list entire publication names just key terms and ideas to inform the summary."
)

mentee_instructions = (
    "Based on the following text, generate a summary paragraph that includes the following information about the individual: "
    "1. Name "
    "2. Educational background "
    "3. Main research interests and goals "
    "4. Notable projects or achievements "
    "5. Any other relevant details that highlight their potential for mentorship and collaboration. "
    "The summary should be concise and informative, making it easy to understand the individual's primary focus and suitability for mentorship."
)

def initialize_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set it in the .env file.")
    return OpenAI(api_key=api_key)

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def prepare_batch_input(data, instructions, column_index):
    return [
        {
            "prompt": f"{instructions}\n{row[column_index]}",
            "temperature": 1.0,
            "max_tokens": 200,
        }
        for _, row in data.iterrows()
    ]

def save_batch_input(batch_input, file_path):
    with open(file_path, "w") as f:
        for item in batch_input:
            f.write(json.dumps(item) + "\n")

def submit_batch_job(client, input_file_path):
    return client.Batch.create(
        input_file=input_file_path, model="gpt-4", output_format="jsonl"
    )

def check_batch_status(client, batch_id):
    status = client.Batch.retrieve(batch_id)
    while status["status"] != "completed":
        time.sleep(30)
        status = client.Batch.retrieve(batch_id)
    return status

def download_batch_results(client, status, output_file_path):
    client.Files.download(status["output_file"], output_file_path)

def process_batch_results(output_file_path):
    summaries = []
    with open(output_file_path, "r") as f:
        for line in f:
            result = json.loads(line)
            summaries.append(result["choices"][0]["text"].strip())
    return summaries

def summarize_cvs(input_file_path, output_file_path):
    client = initialize_openai_client()
    data = load_data(input_file_path)

    mentor_batch_input = prepare_batch_input(data, mentor_instructions, 2)
    mentee_batch_input = prepare_batch_input(data, mentee_instructions, 1)

    mentor_input_file_path = "../simulated_data/mentor_batch_input.jsonl"
    mentee_input_file_path = "../simulated_data/mentee_batch_input.jsonl"
    save_batch_input(mentor_batch_input, mentor_input_file_path)
    save_batch_input(mentee_batch_input, mentee_input_file_path)

    mentor_batch = submit_batch_job(client, mentor_input_file_path)
    mentee_batch = submit_batch_job(client, mentee_input_file_path)

    mentor_status = check_batch_status(client, mentor_batch["id"])
    mentee_status = check_batch_status(client, mentee_batch["id"])

    mentor_output_file_path = "../simulated_data/mentor_batch_output.jsonl"
    mentee_output_file_path = "../simulated_data/mentee_batch_output.jsonl"
    download_batch_results(client, mentor_status, mentor_output_file_path)
    download_batch_results(client, mentee_status, mentee_output_file_path)

    mentor_summaries = process_batch_results(mentor_output_file_path)
    mentee_summaries = process_batch_results(mentee_output_file_path)

    data["Mentor_Summary"] = mentor_summaries
    data["Mentee_Summary"] = mentee_summaries

    data.to_csv(output_file_path, index=False)
    print(f"Summarized CVs saved to {output_file_path}")

def main():
    input_file_path = "../simulated_data/mentor_student_cvs_final.csv"
    output_file_path = "../simulated_data/mentor_student_cvs_with_summaries_final.csv"
    summarize_cvs(input_file_path, output_file_path)

if __name__ == "__main__":
    main()