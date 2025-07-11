import pytest
import os
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# Mock the paths module before importing build_index
@pytest.fixture
def mock_paths_fixture(tmp_path):
    mock_root_dir = tmp_path
    mock_data_dir = mock_root_dir / "data"
    mock_db_dir = mock_root_dir / "db"
    mock_data_dir.mkdir()
    mock_db_dir.mkdir()

    paths_dict = {
        "ROOT_DIR": str(mock_root_dir),
        "PATH_TO_SUMMARY": str(mock_data_dir / "mentor_data_with_summaries.csv"),
        "PATH_TO_MENTOR_DATA": str(mock_data_dir / "mentor_data.csv"),
        "PATH_TO_MENTOR_DATA_RANKED": str(
            mock_data_dir / "mentor_data_summaries_ranks.csv"
        ),
        "PROFESSOR_TYPES_PATH": str(mock_data_dir / "professor_types.txt"),
        "INDEX_SUMMARY_WITH_METADATA": str(mock_db_dir / "index_summary_with_metadata"),
        "INDEX_SUMMARY_ASSISTANT_AND_ABOVE": str(
            mock_db_dir / "index_summary_assistant_and_above"
        ),
        "INDEX_SUMMARY_ABOVE_ASSISTANT": str(
            mock_db_dir / "index_summary_above_assistant"
        ),
    }

    with patch(
        "src.retrieval.build_index.paths", MagicMock(**paths_dict)
    ) as mock_paths:
        yield mock_paths


@patch("src.retrieval.build_index.load_dotenv")
@patch("src.retrieval.build_index.ChatOpenAI")
@patch("src.retrieval.build_index.OpenAIEmbeddings")
@patch("src.retrieval.build_index.FAISS")
@patch("src.retrieval.build_index.pd.read_csv")
@patch("src.retrieval.build_index.os.path.exists")
@patch("builtins.open", new_callable=mock_open)
def test_main_build_index_flow_with_existing_ranked_data(
    mock_open_file,
    mock_os_path_exists,
    mock_read_csv,
    mock_faiss,
    mock_embeddings,
    mock_chat_openai,
    mock_load_dotenv,
    mock_paths_fixture,
):
    # Arrange
    mock_os_path_exists.return_value = True

    ranked_df = pd.DataFrame(
        {
            "Mentor_Data": ["mentor1", "mentor2", "mentor3"],
            "Mentor_Profile": ["profile1", "profile2", "profile3"],
            "Mentor_Summary": ["summary1", "summary2", "summary3"],
            "Professor_Type": [
                "Professor",
                "Associate Professor",
                "Assistant Professor",
            ],
            "Rank": [3, 2, 1],
        }
    )
    mock_read_csv.return_value = ranked_df

    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    mock_faiss.from_texts.return_value = mock_faiss_instance

    # Act
    from src.retrieval.build_index import main

    main()

    # Assert
    mock_load_dotenv.assert_called_once()
    mock_chat_openai.assert_called_once()
    mock_os_path_exists.assert_called_once_with(
        mock_paths_fixture.PATH_TO_MENTOR_DATA_RANKED
    )
    mock_read_csv.assert_called_once_with(
        mock_paths_fixture.PATH_TO_MENTOR_DATA_RANKED, sep="\t"
    )

    mock_embeddings.assert_called_once()
    assert mock_faiss.from_documents.call_count == 1
    assert mock_faiss.from_texts.call_count == 2
    assert mock_faiss_instance.save_local.call_count == 3

    mock_open_file.assert_called_with(mock_paths_fixture.PROFESSOR_TYPES_PATH, "w")
    handle = mock_open_file()
    handle.write.assert_called()
