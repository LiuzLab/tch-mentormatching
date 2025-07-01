import pytest
import os
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock the paths module before importing build_index
@pytest.fixture
def mock_paths(tmp_path):
    mock_root_dir = tmp_path
    mock_data_dir = mock_root_dir / "data"
    mock_db_dir = mock_root_dir / "db"
    mock_data_dir.mkdir()
    mock_db_dir.mkdir()

    with (
        patch('src.config.paths.ROOT_DIR', str(mock_root_dir)),
        patch('src.config.paths.PATH_TO_SUMMARY', str(mock_data_dir / "mentor_data_with_summaries.csv")),
        patch('src.config.paths.PATH_TO_MENTOR_DATA', str(mock_data_dir / "mentor_data.csv")),
        patch('src.config.paths.PATH_TO_MENTOR_DATA_RANKED', str(mock_data_dir / "mentor_data_summaries_ranks.csv")),
        patch('src.config.paths.PROFESSOR_TYPES_PATH', str(mock_data_dir / "professor_types.txt")),
        patch('src.config.paths.INDEX_SUMMARY_WITH_METADATA', str(mock_db_dir / "index_summary_with_metadata")),
        patch('src.config.paths.INDEX_SUMMARY_ASSISTANT_AND_ABOVE', str(mock_db_dir / "index_summary_assistant_and_above")),
        patch('src.config.paths.INDEX_SUMMARY_ABOVE_ASSISTANT', str(mock_db_dir / "index_summary_above_assistant")),
    ):
        yield

@pytest.fixture
def mock_dataframes():
    summary_df = pd.DataFrame({
        "Mentor_Data": ["mentor1", "mentor2", "mentor3"],
        "Mentor_Summary": ["summary1", "summary2", "summary3"]
    })
    mentor_data_df = pd.DataFrame({
        "Mentor_Data": ["mentor1", "mentor2", "mentor3"],
        "Mentor_Profile": ["profile1", "profile2", "profile3"]
    })
    merged_df = pd.DataFrame({
        "Mentor_Data": ["mentor1", "mentor2", "mentor3"],
        "Mentor_Profile": ["profile1", "profile2", "profile3"],
        "Mentor_Summary": ["summary1", "summary2", "summary3"],
        "Professor_Type": ["Professor", "Associate Professor", "Assistant Professor"],
        "Rank": [3, 2, 1]
    })
    return summary_df, mentor_data_df, merged_df

@patch('src.retrieval.build_index.load_dotenv')
@patch('src.retrieval.build_index.ChatOpenAI')
@patch('src.retrieval.build_index.OpenAIEmbeddings')
@patch('src.retrieval.build_index.FAISS')
@patch('src.retrieval.build_index.pd.read_csv')
@patch('src.retrieval.build_index.pd.DataFrame.to_csv')
@patch('src.retrieval.build_index.os.path.exists')
@patch('src.retrieval.build_index.find_professor_type', side_effect=lambda x: {"profile1": "Professor", "profile2": "Associate Professor", "profile3": "Assistant Professor"}.get(x, "Unknown"))
@patch('src.retrieval.build_index.rank_professors', side_effect=lambda df: df.assign(Rank=[3, 2, 1]))
@patch('builtins.open', new_callable=mock_open)
def test_main_build_index_flow(mock_open_file, mock_rank_professors, mock_find_professor_type, mock_os_path_exists, mock_to_csv, mock_read_csv, mock_faiss, mock_embeddings, mock_chat_openai, mock_load_dotenv, mock_paths, mock_dataframes):
    summary_df, mentor_data_df, merged_df = mock_dataframes
    mock_os_path_exists.return_value = False  # Always simulate no existing ranked data to test the full creation flow
    mock_read_csv.side_effect = [summary_df, mentor_data_df] # First call for summary, second for mentor_data

    # Mock FAISS methods
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    mock_faiss.from_texts.return_value = mock_faiss_instance

    # Import main here to ensure load_dotenv is patched before it's called at module level
    from src.retrieval.build_index import main
    main()

    mock_load_dotenv.assert_called_once()
    mock_chat_openai.assert_called_once()
    mock_os_path_exists.assert_called_once_with(mock_paths.PATH_TO_MENTOR_DATA_RANKED)
    
    # Assert read_csv calls for summary and mentor data
    mock_read_csv.assert_any_call(mock_paths.PATH_TO_SUMMARY, sep="\t")
    mock_read_csv.assert_any_call(mock_paths.PATH_TO_MENTOR_DATA)
    assert mock_read_csv.call_count == 2

    mock_find_professor_type.assert_called()
    mock_rank_professors.assert_called()
    mock_to_csv.assert_called_once_with(mock_paths.PATH_TO_MENTOR_DATA_RANKED, sep="\t", index=False)

    # Assert FAISS calls
    mock_embeddings.assert_called()
    assert mock_faiss.from_documents.call_count == 1
    assert mock_faiss.from_texts.call_count == 2
    assert mock_faiss_instance.save_local.call_count == 3
    
    # Assert professor types file creation
    mock_open_file.assert_called_with(mock_paths.PROFESSOR_TYPES_PATH, "w")
    handle = mock_open_file()
    handle.write.assert_called() # Just check if write was called, not specific content
