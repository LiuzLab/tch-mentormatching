import pytest
import os
import pandas as pd
from unittest.mock import patch, AsyncMock
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from main import main as main_pipeline, process_single_mentee
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


@pytest.fixture
def setup_test_environment(tmp_path):
    """Creates a temporary directory structure and patches paths for sandboxing."""
    # Define and create temporary paths
    mentors_dir = tmp_path / "mentors"
    mentees_dir = tmp_path / "mentees"
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "db"
    index_dir = db_dir / "test-embedding-model"
    for d in [mentors_dir, mentees_dir, data_dir, db_dir, index_dir]:
        d.mkdir(exist_ok=True)

    # Create dummy input files with text long enough to pass validation
    mentor1_text = "PRIYA PATEL, Title: Professor. Her research focuses on the application of machine learning to surgical outcomes and developing new AI-driven diagnostic tools. She has extensive experience in Python, TensorFlow, and clinical data analysis. Seeking motivated students."
    mentor2_text = "SOPHIA HALL, Title: Assistant Professor. Her lab works on natural language processing and large language models. They are particularly interested in ethical AI and developing fair and unbiased algorithms. Looking for students with strong programming skills and a passion for NLP."
    (mentors_dir / "mentor1.txt").write_text(mentor1_text)
    (mentors_dir / "mentor2.txt").write_text(mentor2_text)

    # Setup mentee directory
    mentee1_dir = mentees_dir / "mentee1@test.com"
    mentee1_dir.mkdir()
    (mentee1_dir / "mentee1_cv.txt").write_text("A mentee interested in AI and NLP.")
    (mentee1_dir / "mentee1.json").write_text(
        json.dumps(
            {
                "first_name": "Test",
                "last_name": "Mentee",
                "research_Interest": ["AI", "NLP"],
                "submissions_files": ["mentee1_cv.txt"],
            }
        )
    )

    # Patch path variables
    paths_to_patch = {
        "main.PATH_TO_MENTOR_DATA": str(data_dir / "mentor_data.csv"),
        "main.INDEX_SUMMARY_WITH_METADATA": str(index_dir / "faiss_index"),
        "src.retrieval.build_index.paths.INDEX_SUMMARY_WITH_METADATA": str(
            index_dir / "faiss_index"
        ),
    }
    patchers = [patch(p, v) for p, v in paths_to_patch.items()]
    for p in patchers:
        p.start()
    yield {
        "mentors_dir": str(mentors_dir),
        "mentees_dir": str(mentees_dir),
        "data_dir": str(data_dir),
        "mentor_data_path": paths_to_patch["main.PATH_TO_MENTOR_DATA"],
    }
    for p in patchers:
        p.stop()


def mock_openai_embeddings(*args, **kwargs):
    return FakeEmbeddings(size=1)


@pytest.mark.asyncio
@patch("main.summarize_cvs", new_callable=AsyncMock)
@patch("main.OpenAIEmbeddings", mock_openai_embeddings)
@patch("src.retrieval.build_index.OpenAIEmbeddings", mock_openai_embeddings)
async def test_data_pipeline_creates_and_enriches_single_csv(
    mock_summarize_cvs, setup_test_environment
):
    """Tests that the pipeline creates and enriches a single mentor_data.csv."""
    env = setup_test_environment

    # Mock the summarization to add the 'Mentor_Summary' column
    async def mock_summarize_impl(df):
        df["Mentor_Summary"] = "Mocked Summary"
        return df

    mock_summarize_cvs.side_effect = mock_summarize_impl

    # Run the full pipeline
    await main_pipeline(
        mentee_dir=env["mentees_dir"],
        mentor_resume_dir=env["mentors_dir"],
        num_mentors=1,
        overwrite=True,
    )

    # Assert that the single CSV was created and enriched
    mentor_data_path = env["mentor_data_path"]
    assert os.path.exists(mentor_data_path)

    # Check the content of the final CSV
    df = pd.read_csv(mentor_data_path, sep="\t")
    assert "Mentor_Summary" in df.columns
    assert "Professor_Type" in df.columns
    assert "Rank" in df.columns
    assert df.shape[0] == 2  # Two mentors were processed
    assert pd.api.types.is_numeric_dtype(df["Rank"])  # Check for any numeric type


@pytest.mark.asyncio
@patch("main.load_document", return_value="A mentee interested in AI.")
@patch(
    "src.retrieval.search_candidate_mentors.generate_text_async",
    return_value="Mocked mentee summary",
)
@patch("main.evaluate_pair_with_llm", new_callable=AsyncMock)
@patch("main.extract_eval_scores_with_llm", new_callable=AsyncMock)
async def test_matching_logic(
    mock_extract_scores,
    mock_evaluate_pair,
    mock_search_summarize,
    mock_load_document,
    setup_test_environment,
):
    """Tests the matching and evaluation logic with a fake, in-memory FAISS index."""
    env = setup_test_environment
    mentee_cv_path = os.path.join(
        env["mentees_dir"], "mentee1@test.com", "mentee1_cv.txt"
    )

    # Create a fake in-memory vector store
    documents = [
        Document(
            page_content="Mocked Mentor Summary",
            metadata={"Mentor_Profile": "mentor1.txt"},
        )
    ]
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=1))

    mock_evaluate_pair.return_value = "Mocked evaluation text"
    mock_extract_scores.return_value = {"Overall Match Quality": 9.5}

    # Mock mentee data
    mentee_preferences = ["AI", "Machine Learning"]
    mentee_data = {"first_name": "Test", "last_name": "Mentee"}

    # Call the processing function directly
    result = await process_single_mentee(
        mentee_cv_path, vector_store, mentee_preferences, mentee_data, k=1
    )

    # Assert the results
    assert result is not None
    assert result["mentee_name"] == "Test Mentee"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["Criterion Scores"]["Overall Match Quality"] == 9.5
    assert result["mentee_email"] == "mentee1@test.com"
    assert result["mentee_preferences"] == mentee_preferences
