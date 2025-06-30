import argparse
import asyncio
import json
import os
import uuid
import aiofiles
import pandas as pd
import tiktoken
from src.config.client import get_async_openai_client
from src.config.prompts import mentor_instructions, mentee_instructions

def truncate_text(text, max_tokens=3000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = enc.decode(truncated_tokens)
        return truncated_text
    return text

def prepare_batch_input(data, instructions, column_name):
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame with columns: {data.columns}")

    batch_input = []
    for i, row in data.iterrows():
        custom_id = f"request-{uuid.uuid4()}"
        message = truncate_text(row[column_name])
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

async def submit_batch_job(client, input_file_path):
    async with aiofiles.open(input_file_path, "rb") as file:
        batch_input_file = await client.files.create(
            file=await file.read(),
            purpose="batch"
        )
    batch_input_file_id = batch_input_file.id

    return await client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "test batch job"
        }
    )

async def check_batch_status(client, batch_id):
    while True:
        status = await client.batches.retrieve(batch_id)
        print(f"Current batch status: {status.status}")
        if status.status in ["completed", "failed"]:
            break
        await asyncio.sleep(30)
    return status

async def download_batch_results(client, status, output_file_path):
    if hasattr(status, 'output_file_id') and status.output_file_id:
        file_response = await client.files.content(status.output_file_id)
        async with aiofiles.open(output_file_path, 'w') as json_file:
            await json_file.write(file_response.text)
    else:
        if hasattr(status, 'error_file_id') and status.error_file_id:
            error_response = await client.files.content(status.error_file_id)
            async with aiofiles.open(output_file_path.replace('.jsonl', '_error.jsonl'), 'w') as json_file:
                await json_file.write(error_response.text)
            raise ValueError("Batch job failed. Error details saved to the error file.")
        else:
            raise ValueError("Batch job did not produce an output file ID. Check the batch job status and input data.")

async def process_batch_results(file_path):
    summaries = []
    async with aiofiles.open(file_path, 'r') as f:
        async for line in f:
            result = json.loads(line)
            if "response" in result and "body" in result["response"] and "choices" in result["response"]["body"]:
                summary = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                summaries.append(summary)
            else:
                error_info = {
                    "id": result.get("id"),
                    "custom_id": result.get("custom_id"),
                    "error": result.get("error", "No choices key in response body")
                }
                print(f"Error processing result: {error_info}")
                summaries.append("Error: Unable to generate summary for this entry.")
    return summaries

async def summarize_cvs(input_file_path, output_file_path, role="mentor", column_name="Mentor_Data"):
    client = get_async_openai_client()
    data = pd.read_csv(input_file_path)

    instructions = mentor_instructions if role == "mentor" else mentee_instructions
    batch_input = prepare_batch_input(data, instructions, column_name)

    batch_input_file_path = f"../data/{role}_batch_input.jsonl"
    save_batch_input(batch_input, batch_input_file_path)

    batch = await submit_batch_job(client, batch_input_file_path)

    print(f"Batch ID: {batch.id}")

    status = await check_batch_status(client, batch.id)

    if not hasattr(status, 'output_file_id') or not status.output_file_id:
        print(f"Batch details: {status}")
        raise ValueError("Batch job did not produce an output file ID. Check the batch job status and input data.")

    batch_output_file_path = f"../data/{role}_batch_output.jsonl"
    await download_batch_results(client, status, batch_output_file_path)

    summaries = await process_batch_results(batch_output_file_path)

    data[f"{role.capitalize()}_Summary"] = summaries

    data.to_csv(output_file_path, sep='\t', index=False)
    print(f"Summarized CVs saved to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess and summarize documents in batch.")
    parser.add_argument("--in", dest="input_file", required=True, help="Input CSV file to process")
    parser.add_argument("--out", required=True, help="Output CSV file for summaries")
    parser.add_argument(
        "--role", choices=["mentor", "mentee"], default="mentor",
        help="Summary type"
    )
    parser.add_argument(
        "--col", dest="column_name", default="Mentor_Data",
        help="Name of the column to summarize"
    )
    args = parser.parse_args()

    asyncio.run(
        summarize_cvs(
            args.input_file, args.out, args.role, args.column_name
        )
    )

if __name__ == "__main__":
    main()
