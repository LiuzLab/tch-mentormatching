import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.processing.text_utils import truncate_text, summarize_text, async_summarize


def test_truncate_text():
    text = "This is a test text that is longer than the max tokens."
    truncated_text = truncate_text(text, max_tokens=5)
    assert len(truncated_text.split()) <= 5


@patch("src.processing.text_utils.completion_with_backoff")
def test_summarize_text(mock_completion):
    mock_response = {"choices": [{"message": {"content": "This is a summary."}}]}
    mock_completion.return_value = mock_response

    summary = summarize_text("This is a long text to summarize.")
    assert summary == "This is a summary."


@pytest.mark.asyncio
@patch("openai.ChatCompletion.acreate")
async def test_async_summarize(mock_acreate):
    # Mock the async call
    mock_response = {"choices": [{"message": {"content": "This is an async summary."}}]}

    # Since acrate is an async function, the mock needs to be an async mock
    async def async_mock(*args, **kwargs):
        return mock_response

    mock_acreate.return_value = async_mock()

    summary = await async_summarize("This is a long text to summarize asynchronously.")
    assert summary == "This is an async summary."
