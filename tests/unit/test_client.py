import pytest
import os
from unittest.mock import MagicMock, patch
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

@patch('src.config.client.load_dotenv')
@patch('src.config.client.AsyncOpenAI')
@patch('src.config.client.os.getenv')
def test_get_async_openai_client_with_api_key(mock_getenv, mock_async_openai, mock_load_dotenv):
    mock_getenv.return_value = "test_api_key"
    
    # Import the function after patching
    from src.config.client import get_async_openai_client
    
    client = get_async_openai_client()
    
    mock_load_dotenv.assert_called_once()
    mock_getenv.assert_called_once_with("OPENAI_API_KEY")
    mock_async_openai.assert_called_once_with(api_key="test_api_key")
    assert client == mock_async_openai.return_value

@patch('src.config.client.load_dotenv')
@patch('src.config.client.AsyncOpenAI')
@patch('src.config.client.os.getenv')
def test_get_async_openai_client_without_api_key(mock_getenv, mock_async_openai, mock_load_dotenv):
    mock_getenv.return_value = None
    
    # Import the function after patching
    from src.config.client import get_async_openai_client

    with pytest.raises(ValueError, match="API key not found. Please set it in the .env file."):
        get_async_openai_client()
    
    mock_load_dotenv.assert_called_once()
    mock_getenv.assert_called_once_with("OPENAI_API_KEY")
    mock_async_openai.assert_not_called()
