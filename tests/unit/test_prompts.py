import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.prompts import mentor_instructions, mentee_instructions


def test_mentor_instructions_exist():
    assert isinstance(mentor_instructions, str)
    assert "Name" in mentor_instructions
    assert "Institution" in mentor_instructions
    assert "Main research interests" in mentor_instructions


def test_mentee_instructions_exist():
    assert isinstance(mentee_instructions, str)
    assert "Name" in mentee_instructions
    assert "Educational background" in mentee_instructions
    assert "Main research interests and goals" in mentee_instructions
