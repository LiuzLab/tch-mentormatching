import re
import pandas as pd


def extract_and_format_name(mentor_data):
    match = re.match(r"^##\s*(.+?)(?:\n\n|\Z)", mentor_data, re.DOTALL)
    if match:
        name = match.group(1).strip()
        return " ".join(word.capitalize() for word in name.split())
    return "Unknown Name"


def clean_summary(summary):
    # Remove the file identifier and '=====' at the beginning
    cleaned = re.sub(r"^\d+\.txt\s*=+\s*", "", summary)
    return cleaned.strip()


# use this to add a Professor_Type metadata column in the .csv file; allows us to search for
# only professors of a specific typke
import os
from src.config.paths import PROFESSOR_TYPES_PATH


def get_professor_titles():
    """Reads a list of professor titles from the configuration file."""
    if not os.path.exists(PROFESSOR_TYPES_PATH):
        print(
            f"Warning: Professor types file not found at {PROFESSOR_TYPES_PATH}. Using default list."
        )
        return [
            "Chair",
            "Distinguished Professor",
            "Professor",
            "Associate Professor",
            "Assistant Professor",
            "Adjunct Professor",
            "Instructor",
            "Clinical Professor",
        ]
    with open(PROFESSOR_TYPES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


def find_professor_type(mentor_data):
    """
    Finds the professor type from the mentor data text based on a configurable list of titles.
    It uses a more robust regex to find the title in the cleaned text.
    """
    # Regex to find the title, assuming it follows "Title" and precedes "Institution"
    title_match = re.search(r"Title\s+(.*?)\s+Institution", mentor_data, re.IGNORECASE)

    if title_match:
        title_text = title_match.group(1).strip().lower()
        professor_titles = get_professor_titles()

        # Sort titles by length (descending) to match more specific titles first
        # (e.g., "Associate Professor" before "Professor")
        for title in sorted(professor_titles, key=len, reverse=True):
            if title.lower() in title_text:
                return title

    return "Unknown"


def rank_professors(df, professor_type_column="Professor_Type", rank_column="Rank"):
    # Define the ranking dictionary
    rank_mapping = {
        "Chair": 5,
        "Distinguished Professor": 4,
        "Professor": 3,
        "Associate Professor": 2,
        "Assistant Professor": 1,
        "Adjunct Professor": -1,
    }

    # Function to assign rank based on Professor Type
    def assign_rank(professor_type):
        return rank_mapping.get(professor_type, -2)  # Default to -2 for Unknown types

    # Apply the ranking
    df[rank_column] = df[professor_type_column].apply(assign_rank)

    return df
