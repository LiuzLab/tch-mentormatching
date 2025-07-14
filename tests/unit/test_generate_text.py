import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.generate_text import generate_text_async, completion_with_backoff_async
from src.config.model import LLM_MODEL


# Mock the tenacity decorators
@pytest.fixture(autouse=True)
def mock_tenacity_decorators():
    with (
        patch("src.generate_text.retry", lambda *args, **kwargs: lambda fn: fn),
        patch("src.generate_text.stop_after_attempt", MagicMock()),
        patch("src.generate_text.wait_random_exponential", MagicMock()),
    ):
        yield


@pytest.fixture
def mock_openai_client():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    yield mock_client


@pytest.mark.asyncio
async def test_completion_with_backoff_async(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = "Mocked OpenAI Response"

    kwargs = {"model": LLM_MODEL, "messages": [{"role": "user", "content": "test"}]}
    response = await completion_with_backoff_async(mock_openai_client, **kwargs)

    mock_openai_client.chat.completions.create.assert_called_once_with(**kwargs)
    assert response == "Mocked OpenAI Response"


@pytest.mark.asyncio
async def test_generate_text_async(mock_openai_client):
    mock_response_object = MagicMock()
    mock_response_object.choices = [
        MagicMock(message=MagicMock(content="Generated Summary"))
    ]
    mock_openai_client.chat.completions.create.return_value = mock_response_object

    text_input = "This is some input text."
    instructions_input = "Summarize the following:"

    expected_system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant skilled at summarizing and extracting key information from text.",
    }
    expected_user_message = {
        "role": "user",
        "content": f"{instructions_input}\n{text_input}",
    }
    expected_messages = [expected_system_prompt, expected_user_message]

    summary = await generate_text_async(
        mock_openai_client, text_input, instructions_input
    )

    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=LLM_MODEL, messages=expected_messages, temperature=1.0
    )
    assert summary == "Generated Summary"
