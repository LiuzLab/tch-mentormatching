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
    """Creates a temporary directory structure and patches all file paths for sandboxing."""
    # Define and create temporary paths
    mentors_dir = tmp_path / "mentors"
    mentees_dir = tmp_path / "mentees"
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "db"
    index_dir = db_dir / "test-embedding-model"
    for d in [mentors_dir, mentees_dir, output_dir, data_dir, db_dir, index_dir]:
        d.mkdir(exist_ok=True)

    # Create dummy input files with text long enough to pass validation
    mentor1_text = "PRIYA PATEL Title: Assistant Professor. Her research focuses on the application of machine learning to surgical outcomes and developing new AI-driven diagnostic tools. She has extensive experience in Python, TensorFlow, and clinical data analysis."
    mentor2_text = "SOPHIA HALL Title: Professor. Her lab works on natural language processing and large language models. They are particularly interested in ethical AI and developing fair and unbiased algorithms. Looking for students with strong programming skills."
    (mentors_dir / "mentor1.txt").write_text(mentor1_text)
    (mentors_dir / "mentor2.txt").write_text(mentor2_text)
    (mentees_dir / "mentee1.txt").write_text("A mentee interested in AI.")

    # Patch all path variables
    paths_to_patch = {
        'main.PATH_TO_MENTOR_DATA': str(data_dir / "mentor_data.csv"),
        'main.PATH_TO_SUMMARY': str(data_dir / "mentor_data_with_summaries.csv"),
        'main.PATH_TO_MENTOR_DATA_RANKED': str(data_dir / "mentor_data_summaries_ranks.csv"),
        'main.INDEX_SUMMARY_WITH_METADATA': str(index_dir / "index_summary_with_metadata"),
        'main.ROOT_DIR': str(tmp_path),
        'src.retrieval.build_index.paths.PATH_TO_SUMMARY': str(data_dir / "mentor_data_with_summaries.csv"),
        'src.retrieval.build_index.paths.PATH_TO_MENTOR_DATA_RANKED': str(data_dir / "mentor_data_summaries_ranks.csv"),
        'src.retrieval.build_index.paths.INDEX_SUMMARY_WITH_METADATA':
            str(index_dir / "index_summary_with_metadata"),
        'src.retrieval.build_index.paths.INDEX_SUMMARY_ASSISTANT_AND_ABOVE':
            str(index_dir / "index_summary_assistant_and_above"),
        'src.retrieval.build_index.paths.INDEX_SUMMARY_ABOVE_ASSISTANT':
            str(index_dir / "index_summary_above_assistant"),

    }
    patchers = [patch(p, v) for p, v in paths_to_patch.items()]
    for p in patchers: p.start()
    yield {
        "mentors_dir": str(mentors_dir),
        "mentees_dir": str(mentees_dir),
        "data_dir":    str(data_dir),
        "db_dir":      str(db_dir),
    }
    for p in patchers: p.stop()


def mock_openai_embeddings(*args, **kwargs):
    return FakeEmbeddings(size=1)

@pytest.mark.asyncio
@patch("main.summarize_cvs", new_callable=AsyncMock)
@patch("src.retrieval.build_index.find_professor_type", return_value="Professor")
@patch("src.retrieval.build_index.OpenAIEmbeddings", mock_openai_embeddings)
@patch("main.OpenAIEmbeddings", mock_openai_embeddings)
async def test_data_pipeline_creates_files(
    mock_find_professor_type, mock_summarize_cvs, setup_test_environment
):
    """Tests that the data processing pipeline creates all the necessary intermediate files."""
    env = setup_test_environment

    async def mock_summarize_impl(input_path, output_path):
        df = pd.read_csv(input_path)
        df["Mentor_Summary"] = "Mocked Summary"
        df.to_csv(output_path, index=False, sep="\t")
    mock_summarize_cvs.side_effect = mock_summarize_impl

    # Run the pipeline up to the point of matching
    await main_pipeline(
        mentee_dir=env["mentees_dir"],
        mentor_resume_dir=env["mentors_dir"],
        num_mentors=1,
        overwrite=True,
    )

    # Assert that the key data files were created in the temp directory
    assert os.path.exists(os.path.join(env["data_dir"], "mentor_data.csv"))
    assert os.path.exists(os.path.join(env["data_dir"], "mentor_data_with_summaries.csv"))
    assert os.path.exists(os.path.join(env["data_dir"], "mentor_data_summaries_ranks.csv"))

@pytest.mark.asyncio
@patch("main.load_document", return_value="A mentee interested in AI.")
@patch("src.retrieval.search_candidate_mentors.generate_text_async", return_value="Mocked mentee summary")
@patch("main.evaluate_pair_with_llm", new_callable=AsyncMock)
@patch("main.extract_eval_scores_with_llm", new_callable=AsyncMock)
async def test_matching_logic(
    mock_extract_scores, mock_evaluate_pair, mock_search_summarize, mock_load_document, setup_test_environment
):
    """Tests the matching and evaluation logic with a fake, in-memory FAISS index."""
    env = setup_test_environment
    mentee_cv_path = os.path.join(env["mentees_dir"], "mentee1.txt")

    # Create a fake in-memory vector store
    documents = [Document(page_content="Mocked Mentor Summary", metadata={"Mentor_Profile": "mentor1.txt"})]
    vector_store = FAISS.from_documents(documents, FakeEmbeddings(size=1))

    mock_evaluate_pair.return_value = "Mocked evaluation text"
    mock_extract_scores.return_value = {"Overall Match Quality": 9.5}

    # Call the processing function directly
    result = await process_single_mentee(mentee_cv_path, vector_store, k=1)

    # Assert the results
    assert result is not None
    assert result["mentee_name"] == "mentee1"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["Criterion Scores"]["Overall Match Quality"] == 9.5
