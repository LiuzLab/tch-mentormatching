import os
from src.config.model import EMBEDDING_MODEL

# Project root directory
# Assumes this file is in src/config/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Primary Data Paths ---
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_DIR = os.path.join(ROOT_DIR, "db")

# The single, canonical CSV file for all mentor data.
# This file is progressively enriched by the pipeline.
PATH_TO_MENTOR_DATA = os.path.join(DATA_DIR, "mentor_data.csv")

# --- FAISS Index Path ---
# The path is dynamic based on the embedding model to avoid mismatches.
INDEX_DIR = os.path.join(DB_DIR, EMBEDDING_MODEL)
os.makedirs(INDEX_DIR, exist_ok=True)

# The primary FAISS index used for matching.
INDEX_SUMMARY_WITH_METADATA = os.path.join(INDEX_DIR, "faiss_index")
