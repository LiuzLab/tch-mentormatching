import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

    await process_resumes_to_csv(resume_dir, output_csv)

    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)
    assert len(df) == 2
    assert df["Mentor_Profile"][0] == "resume1.txt"
    assert df["Mentor_Data"][0] == "This is resume 1."
    assert df["Mentor_Profile"][1] == "resume2.txt"
    assert df["Mentor_Data"][1] == "This is resume 2."
