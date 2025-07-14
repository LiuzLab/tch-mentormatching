import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from ..config.prompts import mentor_instructions, mentee_instructions
from ..config.model import LLM_MODEL
import tiktoken


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    """Retry-wrapped call to OpenAI ChatCompletion.create"""
    return openai.ChatCompletion.create(**kwargs)


def truncate_text(text: str, max_tokens: int = 3000) -> str:
    """Truncate text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(LLM_MODEL)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return enc.decode(truncated_tokens)
    return text


def summarize_text(text: str, role: str = "mentor") -> str:
    """
    Synchronous summarization of a single text block.
    """
    instructions = mentor_instructions if role == "mentor" else mentee_instructions
    response = completion_with_backoff(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": f"{instructions}\n\n{text}"}],
    )
    return response["choices"][0]["message"]["content"]


async def async_summarize(text: str, role: str = "mentor") -> str:
    """
    Asynchronous summarization of a single text block.
    """
    instructions = mentor_instructions if role == "mentor" else mentee_instructions
    response = await openai.ChatCompletion.acreate(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": f"{instructions}\n\n{text}"}],
    )
    return response["choices"][0]["message"]["content"]
