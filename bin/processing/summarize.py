import openai
import asyncio
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
from config.prompts import mentor_summary_prompt, mentee_summary_prompt
from config.models import DEFAULT_CHAT_MODEL

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def truncate_text(text, max_tokens=3000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = enc.decode(truncated_tokens)
        return truncated_text
    return text

def summarize_text(text, role="mentor"):
    prompt = mentor_summary_prompt if role == "mentor" else mentee_summary_prompt
    response = completion_with_backoff(
        model=DEFAULT_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt + "\n\n" + text}],
    )
    return response["choices"][0]["message"]["content"]

async def async_summarize(session, text, role="mentor"):
    prompt = mentor_summary_prompt if role == "mentor" else mentee_summary_prompt
    response = await session.acreate(
        model=DEFAULT_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt + "\n\n" + text}]
    )
    return response["choices"][0]["message"]["content"]
