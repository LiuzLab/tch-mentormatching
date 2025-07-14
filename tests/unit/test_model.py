import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.model import LLM_MODEL, EVAL_MODEL

SUPPORTED_MODELS = [
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
]


def test_llm_model_valid():
    assert isinstance(LLM_MODEL, str)
    assert (
        LLM_MODEL in SUPPORTED_MODELS
    ), f"LLM_MODEL '{LLM_MODEL}' is not in supported models"


def test_eval_model_valid():
    assert isinstance(EVAL_MODEL, str)
    assert (
        EVAL_MODEL in SUPPORTED_MODELS
    ), f"EVAL_MODEL '{EVAL_MODEL}' is not in supported models"
