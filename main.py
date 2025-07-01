import os
import asyncio
import pandas as pd
from src.processing.io_utils import load_documents
from src.processing.batch import summarize_cvs
from src.retrieval.build_index import main as build_index
from src.retrieval.search_candidate_mentors import search_candidate_mentors
from src.eval.evaluate_matches import evaluate_pair_with_llm, extract_eval_scores_with_llm
from src.eval.html_table_generator import create_mentor_table_html_and_csv_data
from src.config.paths import (
    PATH_TO_MENTOR_DATA,
    PATH_TO_SUMMARY,
    INDEX_SUMMARY_WITH_METADATA,
    ROOT_DIR,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import argparse

async def process_resumes_to_csv(input_dir, output_csv):
    """
    Processes resume files (PDF, DOCX, TXT) in a directory, extracts the text,
    and saves it to a CSV file.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    docs = load_documents(input_dir)
    df = pd.DataFrame(docs, columns=["Mentor_Profile", "Mentor_Data"])
    df.to_csv(output_csv, index=False)
    print(f"Successfully created CSV file at: {output_csv}")


async def process_single_mentee(mentee_cv_path, vector_store):
    # Step 5: Read the mentee CV
    with open(mentee_cv_path, 'r') as f:
        mentee_cv_text = f.read()

    # Step 6: Search for candidate mentors
    search_results = await search_candidate_mentors(
        k=10,
        mentee_cv_text=mentee_cv_text,
        vector_store=vector_store
    )

    # Step 7: Evaluate the matches
    evaluated_matches = []
    for candidate, score in search_results["candidates"]:
        mentor_summary = candidate.page_content
        mentee_summary = search_results["mentee_cv_summary"]
        
        evaluation_text = await evaluate_pair_with_llm(
            mentor_summary=mentor_summary,
            mentee_summary=mentee_summary
        )
        
        scores = await extract_eval_scores_with_llm(
            evaluation_text=evaluation_text
        )
        
        evaluated_matches.append({
            "Mentor Summary": mentor_summary,
            "Similarity Score": score,
            "Criterion Scores": scores,
            "metadata": candidate.metadata
        })

    # Step 8: Generate the HTML table
    html_table, csv_data = create_mentor_table_html_and_csv_data(evaluated_matches)

    # Step 9: Save the HTML table and CSV data
    mentee_name = os.path.splitext(os.path.basename(mentee_cv_path))[0]
    output_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    html_output_path = os.path.join(output_dir, f"{mentee_name}_matches.html")
    csv_output_path = os.path.join(output_dir, f"{mentee_name}_matches.csv")

    with open(html_output_path, "w") as f:
        f.write(html_table)
    
    pd.DataFrame(csv_data).to_csv(csv_output_path, index=False)

    print(f"Results for {mentee_name} saved to {html_output_path} and {csv_output_path}")


async def main(mentee_dir, mentor_resume_dir):
    # Step 1: Process mentor resumes into a CSV file
    await process_resumes_to_csv(mentor_resume_dir, PATH_TO_MENTOR_DATA)

    # Step 2: Summarize the mentor data
    await summarize_cvs(PATH_TO_MENTOR_DATA, PATH_TO_SUMMARY)

    # Step 3: Build the FAISS index
    build_index()

    # Step 4: Load the FAISS index
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(INDEX_SUMMARY_WITH_METADATA, embeddings, allow_dangerous_deserialization=True)

    for mentee_filename in os.listdir(mentee_dir):
        if mentee_filename.endswith((".pdf", ".docx", ".txt")):
            mentee_cv_path = os.path.join(mentee_dir, mentee_filename)
            await process_single_mentee(mentee_cv_path, vector_store)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mentor Matching Pipeline")
    parser.add_argument("--mentees", required=True, help="Path to the directory containing mentee CVs.")
    parser.add_argument("--mentors", required=True, help="Path to the directory containing mentor resumes.")
    args = parser.parse_args()

    asyncio.run(main(args.mentees, args.mentors))
