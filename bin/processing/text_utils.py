# src/preprocessing/text_utils.py
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from config.prompts import mentor_summary_prompt, mentee_summary_prompt
from config.models import DEFAULT_CHAT_MODEL

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    """Retry-wrapped call to OpenAI ChatCompletion.create"""
    return openai.ChatCompletion.create(**kwargs)


def truncate_text(text: str, max_tokens: int = 3000) -> str:
    """Simple truncation: approx. 4 chars per token"""
    return text[: max_tokens * 4]


def summarize_text(text: str, role: str = "mentor") -> str:
    """
    Synchronous summarization of a single text block.
    """
    prompt = mentor_summary_prompt if role == "mentor" else mentee_summary_prompt
    response = completion_with_backoff(
        model=DEFAULT_CHAT_MODEL,
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
    )
    return response["choices"][0]["message"]["content"]


async def async_summarize(text: str, role: str = "mentor") -> str:
    """
    Asynchronous summarization of a single text block.
    """
    prompt = mentor_summary_prompt if role == "mentor" else mentee_summary_prompt
    response = await openai.ChatCompletion.acreate(
        model=DEFAULT_CHAT_MODEL,
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}],
    )
    return response["choices"][0]["message"]["content"]
