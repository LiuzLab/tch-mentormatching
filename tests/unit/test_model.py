import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config.model import LLM_MODEL, EVAL_MODEL

def test_llm_model_constant():
    assert isinstance(LLM_MODEL, str)
    assert LLM_MODEL == "gpt-3.5-turbo-0125"

def test_eval_model_constant():
    assert isinstance(EVAL_MODEL, str)
    assert EVAL_MODEL == "gpt-4"
