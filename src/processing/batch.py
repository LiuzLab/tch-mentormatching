import asyncio
import json
import os
import uuid
import aiofiles
import pandas as pd
import tiktoken
from src.config.paths import ROOT_DIR
from src.config.client import get_async_openai_client
from src.config.prompts import mentor_instructions
from src.config.model import LLM_MODEL


def truncate_text(text, max_tokens=3000):
    """Truncates text to a maximum number of tokens."""
    if not isinstance(text, str):
        return ""
    enc = tiktoken.encoding_for_model(LLM_MODEL)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return enc.decode(truncated_tokens)
    return text


def prepare_batch_input(data):
    """Prepares a list of batch requests for the OpenAI API."""
    batch_input = []
    for i, row in data.iterrows():
        custom_id = f"request-{uuid.uuid4()}"
        message = truncate_text(row["Mentor_Data"])
        body = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{mentor_instructions}\n{message}"},
            ],
            "max_tokens": 1000,
        }
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        batch_input.append(request)
    return batch_input


async def submit_and_wait_for_batch(client, batch_input):
    """Submits a batch job and waits for its completion."""
    # Save batch input to a temporary file
    input_file_path = os.path.join(ROOT_DIR, "data", "mentor_batch_input.jsonl")
    with open(input_file_path, "w") as f:
        for item in batch_input:
            f.write(json.dumps(item) + "\n")

    # Upload the file
    batch_input_file = await client.files.create(
        file=aiofiles.open(input_file_path, "rb"), purpose="batch"
    )

    # Create the batch job
    batch = await client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Submitted batch job with ID: {batch.id}")

    # Wait for the batch to complete
    while True:
        status = await client.batches.retrieve(batch.id)
        print(f"Current batch status: {status.status}")
        if status.status in ["completed", "failed", "cancelled"]:
            break
        await asyncio.sleep(30)

    os.remove(input_file_path)  # Clean up temp input file
    return status


async def get_batch_results(client, status):
    """Downloads and processes batch results."""
    if status.status != "completed" or not status.output_file_id:
        raise ValueError(f"Batch job failed or was cancelled. Status: {status.status}")

    file_response = await client.files.content(status.output_file_id)

    summaries = []
    results_data = file_response.text.strip().split("\n")
    for line in results_data:
        result = json.loads(line)
        if (
            "response" in result
            and "body" in result["response"]
            and "choices" in result["response"]["body"]
        ):
            summary = result["response"]["body"]["choices"][0]["message"][
                "content"
            ].strip()
            summaries.append(summary)
        else:
            summaries.append("Error: Unable to generate summary.")
    return summaries


async def summarize_cvs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'Mentor_Summary' column to the DataFrame by summarizing 'Mentor_Data'.
    """
    client = get_async_openai_client()

    batch_input = prepare_batch_input(df)

    status = await submit_and_wait_for_batch(client, batch_input)

    summaries = await get_batch_results(client, status)

    if len(summaries) != len(df):
        raise ValueError(
            f"Number of summaries ({len(summaries)}) does not match number of mentors ({len(df)})."
        )

    df["Mentor_Summary"] = summaries
    return df
