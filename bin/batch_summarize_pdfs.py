import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import json
import time
from generate_text import generate_text

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

def main():

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

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


    # Prepare the data for batch processing
    mentor_batch_input = []
    mentee_batch_input = []
    for _, row in data.iterrows():
        mentor_batch_input.append(
            {
                "prompt": f"{mentor_instructions}\n{row[2]}",  # Assuming text to summarize is in the third column (index 2)
                "temperature": 1.0,
                "max_tokens": 200,
            }
        )
        mentee_batch_input.append(
            {
                "prompt": f"{mentee_instructions}\n{row[1]}",  # Assuming text to summarize is in the second column (index 1)
                "temperature": 1.0,
                "max_tokens": 200,
            }
        )

    # Save the batch input to files
    mentor_input_file_path = "../simulated_data/mentor_batch_input.jsonl"
    mentee_input_file_path = "../simulated_data/mentee_batch_input.jsonl"
    with open(mentor_input_file_path, "w") as f:
        for item in mentor_batch_input:
            f.write(json.dumps(item) + "\n")
    with open(mentee_input_file_path, "w") as f:
        for item in mentee_batch_input:
            f.write(json.dumps(item) + "\n")

    # Submit batch jobs
    mentor_batch = client.Batch.create(
        input_file=mentor_input_file_path, model="gpt-4", output_format="jsonl"
    )

    mentee_batch = client.Batch.create(
        input_file=mentee_input_file_path, model="gpt-4", output_format="jsonl"
    )

    # Wait for the batch jobs to complete
    mentor_batch_id = mentor_batch["id"]
    mentee_batch_id = mentee_batch["id"]


    def check_batch_status(batch_id):
        status = client.Batch.retrieve(batch_id)
        while status["status"] != "completed":
            time.sleep(30)
            status = client.Batch.retrieve(batch_id)
        return status


    mentor_status = check_batch_status(mentor_batch_id)
    mentee_status = check_batch_status(mentee_batch_id)

    # Download the batch results
    mentor_output_file_path = "../simulated_data/mentor_batch_output.jsonl"
    mentee_output_file_path = "../simulated_data/mentee_batch_output.jsonl"
    client.Files.download(mentor_status["output_file"], mentor_output_file_path)
    client.Files.download(mentee_status["output_file"], mentee_output_file_path)

    # Process the batch results
    mentor_summaries = []
    mentee_summaries = []
    with open(mentor_output_file_path, "r") as f:
        for line in f:
            result = json.loads(line)
            mentor_summaries.append(result["choices"][0]["text"].strip())

    with open(mentee_output_file_path, "r") as f:
        for line in f:
            result = json.loads(line)
            mentee_summaries.append(result["choices"][0]["text"].strip())

    # Add summaries to the dataframe
    data["Mentor_Summary"] = mentor_summaries
    data["Mentee_Summary"] = mentee_summaries

    # Save the results to a new CSV file
    output_file_path = "../simulated_data/mentor_student_cvs_with_summaries_final.csv"
    data.to_csv(output_file_path, index=False)

    print(f"Summarized CVs saved to {output_file_path}")


if __name__ == "__main__":
    main()