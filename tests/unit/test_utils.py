import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import (
    extract_and_format_name,
    clean_summary,
    find_professor_type,
    rank_professors,
)


def test_extract_and_format_name():
    mentor_data = "## john doe\n\nTitle|Professor"
    assert extract_and_format_name(mentor_data) == "John Doe"


def test_clean_summary():
    summary = "12345.txt=====This is a summary."
    assert clean_summary(summary) == "This is a summary."


def test_find_professor_type():
    # Test cases now reflect the new regex which looks for "Title ... Institution"
    mentor_data_professor = (
        "Some text before Title Professor of Engineering Institution some text after"
    )
    mentor_data_associate = "Title Associate Professor Institution"
    mentor_data_assistant = "Title Assistant Professor of Practice Institution"
    mentor_data_unknown = "This text has no title information."

    assert find_professor_type(mentor_data_professor) == "Professor"
    assert find_professor_type(mentor_data_associate) == "Associate Professor"
    assert find_professor_type(mentor_data_assistant) == "Assistant Professor"
    assert find_professor_type(mentor_data_unknown) == "Unknown"


def test_rank_professors():
    data = {
        "Professor_Type": [
            "Professor",
            "Associate Professor",
            "Assistant Professor",
            "Adjunct Professor",
            "Chair",
        ]
    }
    df = pd.DataFrame(data)
    ranked_df = rank_professors(df)
    assert ranked_df["Rank"].tolist() == [3, 2, 1, -1, 5]
