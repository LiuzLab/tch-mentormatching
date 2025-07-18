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
    # Use text that is long enough to pass the clean_and_validate_text check
    long_text = "This is a sufficiently long test text file that is intended to be over one hundred characters to ensure that it passes the minimum length validation check implemented in the text cleaning utility."
    (test_dir / "test.txt").write_text(long_text)
    return str(test_dir), long_text


def test_extract_text_from_txt(setup_test_data):
    test_dir, long_text = setup_test_data
    test_file = os.path.join(test_dir, "test.txt")
    text = extract_text_from_txt(test_file)
    assert text == long_text


def test_load_documents(setup_test_data):
    test_dir, long_text = setup_test_data
    docs = load_documents(test_dir, extensions=[".txt"])
    assert len(docs) == 1
    assert docs[0][0] == "test.txt"
    # The text should be cleaned (extra whitespace removed), but otherwise the same
    assert docs[0][1] == long_text.strip()
