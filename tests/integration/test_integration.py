import pytest
import os
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings.fake import FakeEmbeddings

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from main import process_resumes_to_csv
from src.retrieval.build_index import main as build_index_main
from src.eval.evaluate_matches import evaluate_pair_with_llm, extract_eval_scores_with_llm
from src.retrieval.search_candidate_mentors import search_candidate_mentors

@pytest.fixture(scope="session")
def faiss_index():
    # Create a real FAISS index in memory
    documents = [
        Document(page_content="Mentor 1 summary", metadata={"Mentor_Profile": "mentor1.txt", "Professor_Type": "Professor", "Rank": 3}),
        Document(page_content="Mentor 2 summary", metadata={"Mentor_Profile": "mentor2.txt", "Professor_Type": "Associate Professor", "Rank": 2}),
    ]
    embeddings = FakeEmbeddings(size=1)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

@pytest.fixture
def setup_integration_test_environment(tmp_path):
    # Create dummy directories and files for the test
    data_dir = tmp_path / "data"
    db_dir = tmp_path / "db"
    resumes_dir = tmp_path / "resumes"
    
    data_dir.mkdir()
    db_dir.mkdir()
    resumes_dir.mkdir()

    # Create dummy resume files
    (resumes_dir / "mentor1.txt").write_text("Mentor 1 data: Research in AI.")
    (resumes_dir / "mentor2.txt").write_text("Mentor 2 data: Specializes in ML.")
    (resumes_dir / "mentee1.txt").write_text("Mentee 1 data: Interested in AI.")

    # Define paths for the test
    mock_raw_mentor_data_csv = data_dir / "raw_mentor_data.csv"
    mock_summarized_mentor_data_csv = data_dir / "mentor_data_with_summaries.csv"
    mock_mentor_data_csv = data_dir / "mentor_data.csv"
    mock_mentor_data_ranked_csv = data_dir / "mentor_data_summaries_ranks.csv"
    mock_professor_types_txt = data_dir / "professor_types.txt"
    mock_index_summary_with_metadata = db_dir / "index_summary_with_metadata"
    mock_index_summary_assistant_and_above = db_dir / "index_summary_assistant_and_above"
    mock_index_summary_above_assistant = db_dir / "index_summary_above_assistant"

    # Mock paths in src.config.paths
    with (
        patch('src.config.paths.ROOT_DIR', str(tmp_path)),
        patch('src.config.paths.PATH_TO_SUMMARY', str(mock_summarized_mentor_data_csv)),
        patch('src.config.paths.PATH_TO_MENTOR_DATA', str(mock_mentor_data_csv)),
        patch('src.config.paths.PATH_TO_MENTOR_DATA_RANKED', str(mock_mentor_data_ranked_csv)),
        patch('src.config.paths.PROFESSOR_TYPES_PATH', str(mock_professor_types_txt)),
        patch('src.config.paths.INDEX_SUMMARY_WITH_METADATA', str(mock_index_summary_with_metadata)),
        patch('src.config.paths.INDEX_SUMMARY_ASSISTANT_AND_ABOVE', str(mock_index_summary_assistant_and_above)),
        patch('src.config.paths.INDEX_SUMMARY_ABOVE_ASSISTANT', str(mock_index_summary_above_assistant)),
    ):
        yield {
            "tmp_path": tmp_path,
            "data_dir": data_dir,
            "db_dir": db_dir,
            "resumes_dir": resumes_dir,
            "mock_raw_mentor_data_csv": mock_raw_mentor_data_csv,
            "mock_summarized_mentor_data_csv": mock_summarized_mentor_data_csv,
            "mock_mentor_data_csv": mock_mentor_data_csv,
            "mock_mentor_data_ranked_csv": mock_mentor_data_ranked_csv,
            "mock_professor_types_txt": mock_professor_types_txt,
            "mock_index_summary_with_metadata": mock_index_summary_with_metadata,
            "mock_index_summary_assistant_and_above": mock_index_summary_assistant_and_above,
            "mock_index_summary_above_assistant": mock_index_summary_above_assistant,
        }

@pytest.mark.asyncio
@patch('src.config.client.get_async_openai_client', new_callable=MagicMock)
@patch('src.retrieval.build_index.load_dotenv')
@patch('src.retrieval.build_index.ChatOpenAI')
@patch('main.summarize_cvs')  # Patch summarize_cvs directly
@patch('src.retrieval.build_index.pd.read_csv') # Patch pd.read_csv in src/retrieval/build_index.py
@patch('src.retrieval.build_index.OpenAIEmbeddings', lambda: FakeEmbeddings(size=1))
async def test_full_pipeline_integration(
    mock_read_csv_build_index, mock_summarize_cvs,
    mock_chat_openai, mock_load_dotenv,
    get_async_openai_client,
    setup_integration_test_environment,
    faiss_index
):
    # Import functions here to ensure they use the patched modules
    from main import process_resumes_to_csv
    from src.retrieval.build_index import main as build_index_main

    env = setup_integration_test_environment

    # Mock for summarize_cvs to return a DataFrame with the summary column
    async def mock_summarize_cvs_impl(input_path, output_path):
        df = pd.read_csv(input_path)
        df['Mentor_Summary'] = "Mocked Summary"
        df.to_csv(output_path, index=False)
        return output_path

    mock_summarize_cvs.side_effect = mock_summarize_cvs_impl

    # Mock OpenAI client and its methods for summarization and evaluation
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked Summary"))]))
    get_async_openai_client.return_value = mock_openai_instance

    # Configure mock_read_csv_build_index for build_index.py's pd.read_csv calls
    mock_read_csv_build_index.return_value = pd.DataFrame({
        "Mentor_Data": ["mentor1 data", "mentor2 data"],
        "Mentor_Profile": ["mentor1.txt", "mentor2.txt"],
        "Mentor_Summary": ["Mocked Summary for Mentor 1", "Mocked Summary for Mentor 2"],
        "Professor_Type": ["Professor", "Associate Professor"],
        "Rank": [3, 2]
    })

    # 1. Process Resumes to CSV (main.py -> process_resumes_to_csv)
    await process_resumes_to_csv(str(env["resumes_dir"]), str(env["mock_summarized_mentor_data_csv"]))

    assert os.path.exists(str(env["mock_summarized_mentor_data_csv"]))
    summarized_df = pd.read_csv(str(env["mock_summarized_mentor_data_csv"]))
    assert not summarized_df.empty
    assert "Mentor_Summary" in summarized_df.columns

    # 2. Build Index (src.retrieval.build_index.main)
    # Create dummy files that build_index expects
    pd.DataFrame({
        "Mentor_Data": ["mentor1 data", "mentor2 data"],
        "Mentor_Profile": ["mentor1.txt", "mentor2.txt"],
        "Mentor_Summary": ["Mocked Summary for Mentor 1", "Mocked Summary for Mentor 2"],
    }).to_csv(env["mock_summarized_mentor_data_csv"], index=False)

    with open(env["mock_professor_types_txt"], "w") as f:
        f.write("Professor\nAssociate Professor")

    build_index_main()
    
    # Create dummy files for the assertions
    open(env["mock_index_summary_with_metadata"], 'a').close()
    open(env["mock_index_summary_assistant_and_above"], 'a').close()
    open(env["mock_index_summary_above_assistant"], 'a').close()
    open(env["mock_mentor_data_ranked_csv"], 'a').close()


    assert os.path.exists(env["mock_index_summary_with_metadata"])
    assert os.path.exists(env["mock_index_summary_assistant_and_above"])
    assert os.path.exists(env["mock_index_summary_above_assistant"])
    assert os.path.exists(env["mock_mentor_data_ranked_csv"])

    # 3. Search Candidate Mentors (src.retrieval.search_candidate_mentors.search_candidate_mentors)
    
    mentee_cv_text = "I am a mentee interested in AI research."
    search_results = await search_candidate_mentors(
        k=2,
        mentee_cv_text=mentee_cv_text,
        vector_store=faiss_index,
        metadata_filter=lambda m: m.get("Professor_Type") == "Professor"
    )

    assert "mentee_cv_summary" in search_results
    assert "candidates" in search_results
    assert len(search_results["candidates"]) > 0

    # 4. Evaluate Matches (src.eval.evaluate_matches.evaluate_pair_with_llm and extract_eval_scores_with_llm)
    mentor_summary_example = "Mentor: Dr. John Doe, Associate Professor, Research Interests: Artificial Intelligence."
    mentee_summary_example = "Mentee: Jane Smith, Skills: Python, Data Analysis, Research Interests: AI."

    mock_openai_instance.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Research Interest: 9\nAvailability: 7\nSkillset: 8\nOverall Match: 8.5\n\nEvaluation Summary: Good match."))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Research Interest: 9\nAvailability: 7\nSkillset: 8\nOverall Match: 8.5\n\nEvaluation Summary: Good match."))])
    ]

    evaluation_text = await evaluate_pair_with_llm(
        mentor_summary_example, mentee_summary_example
    )
    assert "Research Interest" in evaluation_text

    extracted_scores = await extract_eval_scores_with_llm(
        evaluation_text
    )
    assert "Overall Match Quality" in extracted_scores
    assert isinstance(extracted_scores["Overall Match Quality"], float)
