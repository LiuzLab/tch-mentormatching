from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import AsyncOpenAI
import asyncio
import functools


# Retry decorator to handle API rate limits
# deprecated
if False:
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
            client, model="gpt-4", messages=messages, temperature=1.0
        )
        return response.choices[0].message.content

from tenacity import retry, stop_after_attempt, wait_random_exponential

def async_retry(*tenacity_args, **tenacity_kwargs):
    def wrapper(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            retrying_func = retry(*tenacity_args, **tenacity_kwargs)(func)
            return await retrying_func(*args, **kwargs)
        return async_wrapper
    return wrapper

@async_retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=10))
async def completion_with_backoff_async(client, **kwargs):
    return await client.chat.completions.create(**kwargs)

async def generate_text_async(client, text, instructions):
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant skilled at summarizing and extracting key information from text.",
    }
    user_message = {"role": "user", "content": f"{instructions}\n{text}"}
    messages = [system_prompt, user_message]
    
    response = await completion_with_backoff_async(
        client, model="gpt-4", messages=messages, temperature=1.0
    )
    return response.choices[0].message.content
