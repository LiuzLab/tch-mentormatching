from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

# Retry decorator to handle API rate limits
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

def generate_text(client, text, instructions):
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant skilled at summarizing and extracting key information from text.",
    }

    user_message = {"role": "user", "content": f"{instructions}\n{text}"}

    messages = [system_prompt, user_message]
    response = completion_with_backoff(
        client,
        model="gpt-4",
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message.content

