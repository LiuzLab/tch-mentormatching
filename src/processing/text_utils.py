import re
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from ..config.prompts import mentor_instructions, mentee_instructions
from ..config.model import LLM_MODEL
import tiktoken

BOILERPLATE_PATTERNS = [
    r"Download vCard",
    r"Login to edit your profile",
    r"Search Profiles",
    r"Same Department",
    r"Explore VIICTR Profiles",
    r"Home\s+About\s+Help\s+History",
    r"CHENG,\s+LILY",
    r"FERGUSON,\s+SUSANNAH",
    r"GOAD,\s+ASHLEY",
    r"LE,\s+LOUIS",
    r"WAGNER,\s+AMY",
    r"ORIT\s+Pediatrics\s+CRA",
]


def clean_and_validate_text(text: str, min_length: int = 100) -> str | None:
    """
    Cleans text extracted from resumes and validates if it's usable.

    - Replaces multiple whitespace chars (tabs, newlines) with a single space.
    - Removes common boilerplate patterns.
    - Checks if the cleaned text meets a minimum length.

    Returns the cleaned text, or None if it's not valid.
    """
    # Replace multiple whitespace characters with a single space
    cleaned_text = re.sub(r"[\s\t\n]+", " ", text).strip()

    # Remove boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    # Re-strip to remove any leading/trailing spaces left after removing boilerplate
    cleaned_text = cleaned_text.strip()

    # Validate
    if len(cleaned_text) < min_length:
        return None

    return cleaned_text


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
