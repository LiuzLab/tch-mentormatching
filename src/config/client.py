import os
from openai import AsyncOpenAI
from dotenv import load_dotenv


def get_async_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set it in the .env file.")
    return AsyncOpenAI(api_key=api_key)
