import argparse
import asyncio
import json
import os

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config.model import EMBEDDING_MODEL
from src.config.paths import (
    INDEX_SUMMARY_WITH_METADATA,
    PATH_TO_MENTOR_DATA,
    PATH_TO_MENTOR_DATA_RANKED,
    PATH_TO_SUMMARY,
    ROOT_DIR,
)
from src.eval.evaluate_matches import (
    evaluate_pair_with_llm,
    extract_eval_scores_with_llm,
)
from src.processing.batch import summarize_cvs
from src.processing.io_utils import load_document, load_documents
from src.retrieval.build_index import build_index
from src.retrieval.search_candidate_mentors import search_candidate_mentors


async def process_single_mentee(mentee_cv_path, vector_store, k=10):
    """Processes a single mentee CV, finds matches, and returns the results."""
    mentee_cv_text = load_document(mentee_cv_path)
    if not mentee_cv_text:
        print(
            f"Could not read or extract text from mentee CV: {mentee_cv_path}. Skipping."
        )
        return None

    search_results = await search_candidate_mentors(
        k=k, mentee_cv_text=mentee_cv_text, vector_store=vector_store
    )

    evaluated_matches = []
    for candidate, score in search_results["candidates"]:
        mentor_summary = candidate.page_content
        mentee_summary = search_results["mentee_cv_summary"]

        evaluation_text = await evaluate_pair_with_llm(
            mentor_summary=mentor_summary, mentee_summary=mentee_summary
        )

        scores = await extract_eval_scores_with_llm(evaluation_text=evaluation_text)

        evaluated_matches.append(
            {
                "Mentor Summary": mentor_summary,
                "Similarity Score": float(score),
                "Criterion Scores": scores,
                "metadata": candidate.metadata,
            }
        )

    # Sort the matches based on the 'Overall Match Quality' score in descending order
    evaluated_matches.sort(
        key=lambda x: x["Criterion Scores"].get("Overall Match Quality", 0),
        reverse=True,
    )

    mentee_name = os.path.splitext(os.path.basename(mentee_cv_path))[0]
    return {"mentee_name": mentee_name, "matches": evaluated_matches}


async def main(mentee_dir, mentor_resume_dir, num_mentors, overwrite=False):
    # --- Step 1: Process raw mentor resumes into a CSV file ---
    if overwrite or not os.path.exists(PATH_TO_MENTOR_DATA):
        print("Step 1: Processing mentor resumes into CSV...")
        if not os.path.exists(mentor_resume_dir):
            raise FileNotFoundError(
                f"Mentor resume directory not found: {mentor_resume_dir}"
            )

        docs = load_documents(mentor_resume_dir)
        if not docs:
            raise ValueError(
                f"No documents (PDF, DOCX, TXT) found in {mentor_resume_dir}"
            )

        df = pd.DataFrame(docs, columns=["Mentor_Profile", "Mentor_Data"])
        df.to_csv(PATH_TO_MENTOR_DATA, index=False)
        print(f"Successfully created raw mentor data CSV at: {PATH_TO_MENTOR_DATA}")
    else:
        print(
            f"Skipping Step 1: Raw mentor data CSV already exists at {PATH_TO_MENTOR_DATA}"
        )

    # --- Step 2: Summarize the mentor data ---
    if overwrite or not os.path.exists(PATH_TO_SUMMARY):
        print("\nStep 2: Summarizing mentor data...")
        await summarize_cvs(PATH_TO_MENTOR_DATA, PATH_TO_SUMMARY)
    else:
        print(
            f"Skipping Step 2: Summarized mentor data already exists at {PATH_TO_SUMMARY}"
        )

    # --- Step 3: Build the FAISS index and ranked data file ---
    # This step runs if the index itself is missing, ensuring it's created
    # even if the intermediate ranked data file exists.
    if overwrite or not os.path.exists(INDEX_SUMMARY_WITH_METADATA):
        print("\nStep 3: Building FAISS index and ranking mentors...")
        build_index()
    else:
        print(
            f"Skipping Step 3: FAISS index already exists at {INDEX_SUMMARY_WITH_METADATA}"
        )

    # --- Step 4: Load the FAISS index for matching ---
    print("\nLoading FAISS index for matching...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if not os.path.exists(INDEX_SUMMARY_WITH_METADATA):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_SUMMARY_WITH_METADATA}. Please run the script with --overwrite."
        )

    vector_store = FAISS.load_local(
        INDEX_SUMMARY_WITH_METADATA, embeddings, allow_dangerous_deserialization=True
    )

    # --- Step 5: Process each mentee ---
    print("\nProcessing mentees...")
    all_matches = []
    for mentee_filename in os.listdir(mentee_dir):
        if mentee_filename.lower().endswith((".pdf", ".docx", ".txt")):
            mentee_cv_path = os.path.join(mentee_dir, mentee_filename)
            print(f"Processing {mentee_filename}...")
            mentee_results = await process_single_mentee(
                mentee_cv_path, vector_store, num_mentors
            )
            if mentee_results:
                all_matches.append(mentee_results)

    # --- Step 6: Save the final JSON output ---
    output_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, "best_matches.json")

    with open(json_output_path, "w") as f:
        json.dump(all_matches, f, indent=4)

    print(f"\nAll mentee matches saved to {json_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mentor Matching Pipeline")
    parser.add_argument(
        "--mentees", required=True, help="Path to the directory containing mentee CVs."
    )
    parser.add_argument(
        "--mentors",
        required=True,
        help="Path to the directory containing mentor resumes.",
    )
    parser.add_argument(
        "--num_mentors",
        type=int,
        required=True,
        help="Number of desired matches (length of table)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached files and re-run the full data processing pipeline.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.mentees, args.mentors, args.num_mentors, args.overwrite))
