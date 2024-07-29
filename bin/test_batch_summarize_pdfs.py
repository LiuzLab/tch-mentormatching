import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json
import time
import tiktoken
import uuid

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

def initialize_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set it in the .env file.")
    return OpenAI(api_key=api_key)

def load_data(file_path, num_rows=10):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path).head(num_rows)
    return data

def truncate_text(text, max_tokens=3000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = enc.decode(truncated_tokens)
        return truncated_text
    return text

def prepare_batch_input(data, instructions, column_index):
    if column_index >= len(data.columns):
        raise IndexError(f"Column index {column_index} is out of bounds for DataFrame with columns: {data.columns}")

    batch_input = []
    for i, row in data.iterrows():
        custom_id = f"request-{uuid.uuid4()}"
        message = truncate_text(row[column_index])
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{instructions}\n{message}"}
            ],
            "max_tokens": 1000,
        }
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        batch_input.append(request)

    return batch_input

def save_batch_input(batch_input, file_path):
    with open(file_path, "w") as f:
        for item in batch_input:
            f.write(json.dumps(item) + "\n")

def submit_batch_job(client, input_file_path):
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    return client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "test batch job"
        }
    )

def check_batch_status(client, batch_id):
    while True:
        status = client.batches.retrieve(batch_id)
        print(f"Current batch status: {status.status}")
        if status.status in ["completed", "failed"]:
            break
        time.sleep(30)
    return status

def download_batch_results(client, status, output_file_path, mentor_batch):
    if hasattr(status, 'output_file_id') and status.output_file_id:
        file_response = client.files.content(status.output_file_id)
        with open(output_file_path, 'w') as json_file:
            json_file.write(file_response.text)
    else:
        if hasattr(status, 'error_file_id') and status.error_file_id:
            error_response = client.files.content(status.error_file_id)
            with open(output_file_path.replace('.jsonl', '_error.jsonl'), 'w') as json_file:
                json_file.write(error_response.text)
            raise ValueError("Batch job failed. Error details saved to the error file.")
        else:
            raise ValueError("Batch job did not produce an output file ID. Check the batch job status and input data.")

def process_batch_results(file_path):
    summaries = []
    with open(file_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            if "response" in result and "body" in result["response"] and "choices" in result["response"]["body"]:
                summary = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                summaries.append(summary)
            else:
                # Handle cases where the response does not have the expected structure
                error_info = {
                    "id": result.get("id"),
                    "custom_id": result.get("custom_id"),
                    "error": result.get("error", "No choices key in response body")
                }
                print(f"Error processing result: {error_info}")
                summaries.append("Error: Unable to generate summary for this entry.")
    return summaries

def summarize_cvs(input_file_path, output_file_path, num_rows=10):
    client = initialize_openai_client()
    data = load_data(input_file_path, num_rows)

    mentor_batch_input = prepare_batch_input(data, mentor_instructions, 0)  # Change column_index to 0

    mentor_input_file_path = "../data/mentor_batch_input_test.jsonl"
    save_batch_input(mentor_batch_input, mentor_input_file_path)

    mentor_batch = submit_batch_job(client, mentor_input_file_path)
    
    print(f"Batch ID: {mentor_batch.id}")

    mentor_status = check_batch_status(client, mentor_batch.id)
    
    if not hasattr(mentor_status, 'output_file_id') or not mentor_status.output_file_id:
        print(f"Batch details: {mentor_status}")
        raise ValueError("Batch job did not produce an output file ID. Check the batch job status and input data.")

    mentor_output_file_path = "../data/mentor_batch_output_test.jsonl"
    download_batch_results(client, mentor_status, mentor_output_file_path, mentor_batch)

    mentor_summaries = process_batch_results(mentor_output_file_path)

    data["Mentor_Summary"] = mentor_summaries

    data.to_csv(output_file_path, sep='\t', index=False)
    print(f"Summarized CVs (first {num_rows} rows) saved to {output_file_path}")

def main():
    input_file_path = "../data/mentor_data.csv"
    output_file_path = "../data/mentor_data_with_summaries_test.csv"
    summarize_cvs(input_file_path, output_file_path, num_rows=10)

if __name__ == "__main__":
    main()

