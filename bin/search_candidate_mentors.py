import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .generate_text import generate_text
from .batch_summarize_pdfs import mentee_instructions, initialize_openai_client

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo-0125"

# Initialize OpenAI client
client = initialize_openai_client()

def search_candidate_mentors(k=36, mentee_cv_text="", vector_store=None):
    if vector_store is None:
        raise ValueError("Vector store must be provided")
    
    mentee_cv_summary = generate_text(client, mentee_cv_text, mentee_instructions)
    candidates = vector_store.similarity_search_with_score(mentee_cv_summary, k=k, fetch_k=k)
    return {
        "mentee_cv_summary": mentee_cv_summary,
        "candidates": candidates
    }

if __name__ == "__main__":
    # You can add test code here if needed
    pass