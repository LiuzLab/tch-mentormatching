import sys
import os
import pandas as pd
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.processing.io_utils import (
    extract_text_from_txt,
    load_documents,
)


@pytest.fixture
def setup_test_data(tmp_path):
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    (test_dir / "test.txt").write_text("This is a test text file.")
    return str(test_dir)


def test_extract_text_from_txt(setup_test_data):
    test_file = os.path.join(setup_test_data, "test.txt")
    text = extract_text_from_txt(test_file)
    assert text == "This is a test text file."


def test_load_documents(setup_test_data):
    docs = load_documents(setup_test_data, extensions=[".txt"])
    assert len(docs) == 1
    assert docs[0][0] == "test.txt"
    assert docs[0][1] == "This is a test text file."
