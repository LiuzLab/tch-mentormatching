import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.build_index import build_index
from langchain_core.documents import Document


@pytest.fixture
def mock_paths(tmp_path):
    """Fixture to mock the paths used in the build_index module."""
    with patch("src.retrieval.build_index.paths") as mock_paths_patch:
        mock_paths_patch.INDEX_SUMMARY_WITH_METADATA = str(tmp_path / "test_index")
        yield mock_paths_patch


@pytest.fixture
def sample_mentor_df():
    """Fixture to create a sample mentor DataFrame for testing."""
    return pd.DataFrame(
        {
            "Mentor_Summary": [
                "Summary of a great mentor.",
                "Summary of another mentor.",
            ],
            "Mentor_Profile": ["profile1.pdf", "profile2.pdf"],
            "Professor_Type": ["Professor", "Assistant Professor"],
            "Rank": [3.0, 1.0],
        }
    )


@patch("src.retrieval.build_index.FAISS")
@patch("src.retrieval.build_index.OpenAIEmbeddings")
def test_build_index_creates_and_saves_vector_store(
    mock_openai_embeddings, mock_faiss, sample_mentor_df, mock_paths
):
    """
    Tests that build_index correctly processes a DataFrame, creates Documents,
    initializes an embedding model, and creates and saves a FAISS vector store.
    """
    # Arrange
    mock_embedding_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_embedding_instance

    mock_vector_store_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store_instance

    # Act
    build_index(sample_mentor_df)

    # Assert
    # 1. Check if OpenAIEmbeddings was initialized correctly
    mock_openai_embeddings.assert_called_once()

    # 2. Check if FAISS.from_documents was called
    mock_faiss.from_documents.assert_called_once()

    # 3. Verify the structure of the documents passed to FAISS
    call_args = mock_faiss.from_documents.call_args
    passed_documents = call_args.kwargs["documents"]
    assert len(passed_documents) == 2
    assert isinstance(passed_documents[0], Document)
    assert passed_documents[0].page_content == "Summary of a great mentor."
    assert passed_documents[0].metadata["Rank"] == 3.0
    assert passed_documents[1].page_content == "Summary of another mentor."
    assert passed_documents[1].metadata["Professor_Type"] == "Assistant Professor"

    # 4. Verify the correct embedding model was used
    assert call_args.kwargs["embedding"] == mock_embedding_instance

    # 5. Check if the vector store was saved to the correct path
    mock_vector_store_instance.save_local.assert_called_once_with(
        mock_paths.INDEX_SUMMARY_WITH_METADATA
    )


def test_build_index_raises_error_on_missing_columns():
    """
    Tests that build_index raises a ValueError if the input DataFrame
    is missing any of the required columns.
    """
    # Arrange
    incomplete_df = pd.DataFrame(
        {
            "Mentor_Summary": ["A summary"],
            # Missing "Mentor_Profile", "Professor_Type", "Rank"
        }
    )

    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        build_index(incomplete_df)

    assert "must contain the following columns" in str(excinfo.value)
