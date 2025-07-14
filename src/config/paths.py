import os

# Project root directory
# Assumes the script is in src/config/paths.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File paths
PATH_TO_SUMMARY = os.path.join(ROOT_DIR, "data/mentor_data_with_summaries.csv")
PATH_TO_MENTOR_DATA = os.path.join(ROOT_DIR, "data/mentor_data.csv")
PATH_TO_SUMMARY_DATA = os.path.join(ROOT_DIR, "data/summary_data.csv")
PATH_TO_MENTOR_DATA_RANKED = os.path.join(
    ROOT_DIR, "data/mentor_data_summaries_ranks.csv"
)
PROFESSOR_TYPES_PATH = os.path.join(ROOT_DIR, "data/professor_types.txt")

# FAISS index paths
INDEX_SUMMARY_WITH_METADATA = os.path.join(ROOT_DIR, "db/index_summary_with_metadata")
INDEX_SUMMARY_ASSISTANT_AND_ABOVE = os.path.join(
    ROOT_DIR, "db/index_summary_assistant_and_above"
)
INDEX_SUMMARY_ABOVE_ASSISTANT = os.path.join(
    ROOT_DIR, "db/index_summary_above_assistant"
)
