import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

_client = None


def get_async_openai_client():
    """
    Returns a lazily-initialized, singleton instance of the AsyncOpenAI client.
    The client is only created on the first call to this function.
    """
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # In a test environment, we might not have a key, which is fine
            # as the calls will be mocked.
            print(
                "Warning: OPENAI_API_KEY not found. Proceeding without it for testing."
            )
            # We can't initialize with a None key, so we'll use a dummy key
            # if we're in a test environment. This will be mocked anyway.
            api_key = "test-key-not-real"
        _client = AsyncOpenAI(api_key=api_key)
    return _client
