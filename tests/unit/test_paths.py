import os
import sys
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import paths
from src.config.model import EMBEDDING_MODEL


def test_root_dir_is_correct():
    """Tests that ROOT_DIR is correctly pointing to the project's root."""
    # This assumes the test is run from within the project structure.
    # The project root is two levels up from tests/unit.
    expected_root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    assert paths.ROOT_DIR == expected_root_dir


def test_data_dir_is_correct():
    """Tests that DATA_DIR is correctly constructed."""
    expected_path = os.path.join(paths.ROOT_DIR, "data")
    assert paths.DATA_DIR == expected_path


def test_path_to_mentor_data_is_correct():
    """Tests that PATH_TO_MENTOR_DATA points to the correct file."""
    expected_path = os.path.join(paths.DATA_DIR, "mentor_data.csv")
    assert paths.PATH_TO_MENTOR_DATA == expected_path


def test_index_dir_is_dynamic():
    """Tests that the INDEX_DIR is correctly created based on the embedding model."""
    expected_path = os.path.join(paths.DB_DIR, EMBEDDING_MODEL)
    assert paths.INDEX_DIR == expected_path
    assert os.path.exists(paths.INDEX_DIR)  # Should be created on import


def test_primary_faiss_index_path_is_correct():
    """Tests that INDEX_SUMMARY_WITH_METADATA points to the correct file."""
    expected_path = os.path.join(paths.INDEX_DIR, "faiss_index")
    assert paths.INDEX_SUMMARY_WITH_METADATA == expected_path
