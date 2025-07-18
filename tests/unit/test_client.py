import pytest
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import the module that we will be testing
from src.config import client


@pytest.fixture(autouse=True)
def reset_singleton():
    """Fixture to reset the singleton client instance before each test."""
    client._client = None
    yield


@patch("src.config.client.load_dotenv")
@patch("src.config.client.AsyncOpenAI")
@patch("src.config.client.os.getenv")
def test_get_async_openai_client_with_api_key(
    mock_getenv, mock_async_openai, mock_load_dotenv
):
    mock_getenv.return_value = "test_api_key"

    # First call should initialize the client
    c1 = client.get_async_openai_client()
    mock_load_dotenv.assert_called_once()
    mock_getenv.assert_called_once_with("OPENAI_API_KEY")
    mock_async_openai.assert_called_once_with(api_key="test_api_key")
    assert c1 == mock_async_openai.return_value

    # Second call should return the same instance without re-initializing
    c2 = client.get_async_openai_client()
    mock_load_dotenv.assert_called_once()  # Should still be 1
    assert c1 == c2


@patch("src.config.client.load_dotenv")
@patch("src.config.client.AsyncOpenAI")
@patch("src.config.client.os.getenv")
def test_get_async_openai_client_without_api_key(
    mock_getenv, mock_async_openai, mock_load_dotenv, capsys
):
    mock_getenv.return_value = None

    # Call the function
    client.get_async_openai_client()

    # Check that it initialized with the dummy key
    mock_async_openai.assert_called_once_with(api_key="test-key-not-real")

    # Check that the warning was printed to stdout
    captured = capsys.readouterr()
    assert "Warning: OPENAI_API_KEY not found" in captured.out
