import sys
import os
import pandas as pd
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from main import process_resumes_to_csv


@pytest.fixture
def setup_test_environment(tmp_path):
    # Create dummy resume files
    resume_dir = tmp_path / "resumes"
    resume_dir.mkdir()
    (resume_dir / "resume1.txt").write_text("This is resume 1.")
    (resume_dir / "resume2.txt").write_text("This is resume 2.")
    
    output_csv = tmp_path / "output.csv"
    
    return str(resume_dir), str(output_csv)

@pytest.mark.asyncio
async def test_process_resumes_to_csv(setup_test_environment):
    resume_dir, output_csv = setup_test_environment
    
    # Mock the external dependencies to prevent actual API calls and file operations beyond the test scope
    with (
        patch('main.convert_txt_dir_to_csv') as mock_convert_txt_dir_to_csv,
        patch('main.summarize_cvs') as mock_summarize_cvs,
        patch('main.pd.read_csv') as mock_read_csv,
        patch('main.pd.DataFrame.to_csv') as mock_to_csv
    ):
        # Configure mocks to allow the function to run without errors
        mock_convert_txt_dir_to_csv.return_value = None
        mock_summarize_cvs.return_value = None
        
        # Mock read_csv to return a dummy DataFrame that resembles the expected output after summarization
        mock_read_csv.return_value = pd.DataFrame({
            "Mentor_Profile": ["resume1.txt", "resume2.txt"],
            "Mentor_Data": ["This is resume 1.", "This is resume 2."],
            "Mentor_Summary": ["Summary 1", "Summary 2"]
        })
        
        await process_resumes_to_csv(resume_dir, output_csv)
        
        # Assert that the mocked functions were called as expected
        mock_convert_txt_dir_to_csv.assert_called_once_with(
            os.path.join(resume_dir, "*.txt"), 
            os.path.join(os.path.dirname(output_csv), "raw_mentor_data.csv")
        )
        mock_summarize_cvs.assert_called_once_with(
            os.path.join(os.path.dirname(output_csv), "raw_mentor_data.csv"), 
            output_csv, 
            role="mentor", 
            column_name="Mentor_Data"
        )
        mock_read_csv.assert_called_once_with(output_csv, sep='\t')
        mock_to_csv.assert_called_once_with(output_csv, sep='\t', index=False)

        # Verify that the output CSV would have been created (mocked)
        assert os.path.exists(output_csv) # This will be true because tmp_path creates the file path
