import pytest
import os
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from main import main as main_pipeline
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings.fake import FakeEmbeddings

@pytest.fixture
def setup_test_environment(tmp_path):
    """Creates a temporary directory structure for testing."""
    # Create dummy directories
    mentors_dir = tmp_path / "mentors"
    mentees_dir = tmp_path / "mentees"
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "db"

    for d in [mentors_dir, mentees_dir, output_dir, data_dir, db_dir]:
        d.mkdir()

    # Create dummy files with longer, more realistic text
    mentor1_text = """
    PRIYA PATEL
    Title: Assistant Professor
    Institution: Baylor College of Medicine
    Department: Department of Surgery
    Research Interests: My research focuses on the application of machine learning to surgical outcomes and developing new AI-driven diagnostic tools. I have extensive experience in Python, TensorFlow, and clinical data analysis.
    """
    mentor2_text = """
    SOPHIA HALL
    Title: Professor
    Institution: University of Houston
    Department: Department of Computer Science
    Research Interests: My lab works on natural language processing and large language models. We are particularly interested in ethical AI and developing fair and unbiased algorithms. Looking for students with strong programming skills.
    """
    mentee_text = "I am a prospective mentee interested in the application of AI and machine learning in medicine. I have a background in data analysis and Python."

    (mentors_dir / "mentor1.txt").write_text(mentor1_text)
    (mentors_dir / "mentor2.txt").write_text(mentor2_text)
    (mentees_dir / "mentee1.txt").write_text(mentee_text)

    # Mock paths in src.config.paths
    with patch("src.config.paths.ROOT_DIR", str(tmp_path)):
        yield {
            "mentors_dir": str(mentors_dir),
            "mentees_dir": str(mentees_dir),
            "output_dir": str(output_dir),
            "data_dir": str(data_dir),
            "db_dir": str(db_dir),
        }

@pytest.mark.asyncio
@patch("src.processing.batch.summarize_cvs", new_callable=AsyncMock)
@patch("src.retrieval.build_index.OpenAIEmbeddings", lambda: FakeEmbeddings(size=1))
@patch("main.OpenAIEmbeddings", lambda: FakeEmbeddings(size=1))
@patch("src.retrieval.search_candidate_mentors.generate_text_async", new_callable=AsyncMock)
@patch("main.evaluate_pair_with_llm", new_callable=AsyncMock)
@patch("main.extract_eval_scores_with_llm", new_callable=AsyncMock)
async def test_full_pipeline(
    mock_extract_scores,
    mock_evaluate_pair,
    mock_search_summarize,
    mock_summarize_cvs,
    setup_test_environment,
):
    """
    Tests the full main.py pipeline, mocking external API calls.
    """
    env = setup_test_environment

    # --- Mock Implementations ---
    async def mock_summarize_impl(input_path, output_path):
        df = pd.read_csv(input_path)
        df["Mentor_Summary"] = "Mocked Summary: " + df["Mentor_Data"]
        df.to_csv(output_path, index=False, sep="\t")

    mock_summarize_cvs.side_effect = mock_summarize_impl
    mock_search_summarize.return_value = "Mocked mentee summary"
    mock_evaluate_pair.return_value = "Mocked evaluation text"
    mock_extract_scores.return_value = {
        "Overall Match Quality": 9.0,
        "Research Interest": 9,
        "Availability": 8,
        "Skillset": 10,
        "Evaluation Summary": "Excellent match.",
    }

    # --- Run the Pipeline ---
    await main_pipeline(
        mentee_dir=env["mentees_dir"],
        mentor_resume_dir=env["mentors_dir"],
        num_mentors=2,
        overwrite=True,
    )

    # --- Assertions ---
    # Check if the final JSON output was created
    output_json_path = os.path.join(env["output_dir"], "best_matches.json")
    assert os.path.exists(output_json_path)

    # Check the content of the JSON file
    with open(output_json_path, "r") as f:
        results = json.load(f)

    assert len(results) == 1
    mentee_result = results[0]
    assert mentee_result["mentee_name"] == "mentee1"
    assert len(mentee_result["matches"]) > 0

    first_match = mentee_result["matches"][0]
    assert "Mentor Summary" in first_match
    assert "Criterion Scores" in first_match
    assert first_match["Criterion Scores"]["Overall Match Quality"] == 9.0
