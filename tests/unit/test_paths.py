import os
import sys
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import paths

def test_root_dir():
    # This test assumes that the ROOT_DIR is correctly set to the project root
    # which is two levels up from the src/config directory.
    expected_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    assert paths.ROOT_DIR == expected_root_dir

def test_path_to_summary():
    expected_path = os.path.join(paths.ROOT_DIR, "data/mentor_data_with_summaries.csv")
    assert paths.PATH_TO_SUMMARY == expected_path

def test_path_to_mentor_data():
    expected_path = os.path.join(paths.ROOT_DIR, "data/mentor_data.csv")
    assert paths.PATH_TO_MENTOR_DATA == expected_path

def test_path_to_summary_data():
    expected_path = os.path.join(paths.ROOT_DIR, "data/summary_data.csv")
    assert paths.PATH_TO_SUMMARY_DATA == expected_path

def test_path_to_mentor_data_ranked():
    expected_path = os.path.join(paths.ROOT_DIR, "data/mentor_data_summaries_ranks.csv")
    assert paths.PATH_TO_MENTOR_DATA_RANKED == expected_path

def test_professor_types_path():
    expected_path = os.path.join(paths.ROOT_DIR, "data/professor_types.txt")
    assert paths.PROFESSOR_TYPES_PATH == expected_path

def test_index_summary_with_metadata():
    expected_path = os.path.join(paths.ROOT_DIR, "db/index_summary_with_metadata")
    assert paths.INDEX_SUMMARY_WITH_METADATA == expected_path

def test_index_summary_assistant_and_above():
    expected_path = os.path.join(paths.ROOT_DIR, "db/index_summary_assistant_and_above")
    assert paths.INDEX_SUMMARY_ASSISTANT_AND_ABOVE == expected_path

def test_index_summary_above_assistant():
    expected_path = os.path.join(paths.ROOT_DIR, "db/index_summary_above_assistant")
    assert paths.INDEX_SUMMARY_ABOVE_ASSISTANT == expected_path
