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
from src.utils import find_professor_type, rank_professors


def safe_save_csv(df, path):
    """Saves a DataFrame to a CSV file atomically using tabs as separators."""
    temp_path = path + ".tmp"
    df.to_csv(temp_path, index=False, sep="\t")
    os.replace(temp_path, path)


async def process_single_mentee(
    mentee_cv_path, vector_store, mentee_preferences, mentee_data, k=10
):
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
            mentor_summary=mentor_summary,
            mentee_summary=mentee_summary,
            mentee_preferences=mentee_preferences,
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

    evaluated_matches.sort(
        key=lambda x: x["Criterion Scores"].get("Overall Match Quality", 0),
        reverse=True,
    )

    mentee_name = f"{mentee_data.get('first_name')} {mentee_data.get('last_name')}"
    mentee_email = os.path.basename(os.path.dirname(mentee_cv_path))

    return {
        "mentee_name": mentee_name,
        "mentee_email": mentee_email,
        "mentee_preferences": mentee_preferences,
        "matches": evaluated_matches,
    }


async def main(
    mentee_dir, mentor_resume_dir, num_mentors, overwrite=False, output_dir=None
):
    # --- Step 1: Initial Data Loading ---
    if overwrite or not os.path.exists(PATH_TO_MENTOR_DATA):
        print("Step 1: Processing mentor resumes from source...")
        if not mentor_resume_dir or not os.path.exists(mentor_resume_dir):
            raise ValueError(
                "--mentors directory is required when running with --overwrite or when mentor_data.csv does not exist."
            )
        docs = load_documents(mentor_resume_dir)
        if not docs:
            raise ValueError(
                f"No documents (PDF, DOCX, TXT) found in {mentor_resume_dir}"
            )
        df = pd.DataFrame(docs, columns=["Mentor_Profile", "Mentor_Data"])
        safe_save_csv(df, PATH_TO_MENTOR_DATA)
        print(f"Successfully created raw mentor data CSV at: {PATH_TO_MENTOR_DATA}")
    else:
        print(f"Skipping Step 1: Using existing mentor data at {PATH_TO_MENTOR_DATA}")

    df = pd.read_csv(PATH_TO_MENTOR_DATA, sep="\t")

    # --- Step 2: Summarize Mentor Data ---
    if "Mentor_Summary" not in df.columns:
        print("\nStep 2: Summarizing mentor data...")
        df = await summarize_cvs(df)
        safe_save_csv(df, PATH_TO_MENTOR_DATA)
        print(f"Successfully added summaries to {PATH_TO_MENTOR_DATA}")
    else:
        print("Skipping Step 2: Mentor summaries already exist.")

    # --- Step 3: Rank Mentors ---
    if "Rank" not in df.columns:
        print("\nStep 3: Ranking mentors...")
        df["Professor_Type"] = [
            find_professor_type(text) for text in df["Mentor_Data"].fillna("")
        ]
        df = rank_professors(df)
        safe_save_csv(df, PATH_TO_MENTOR_DATA)
        print(f"Successfully added ranks to {PATH_TO_MENTOR_DATA}")
    else:
        print("Skipping Step 3: Mentor ranks already exist.")

    # --- Step 4: Build FAISS Index ---
    if overwrite or not os.path.exists(INDEX_SUMMARY_WITH_METADATA):
        print("\nStep 4: Building FAISS index...")
        build_index(df)
    else:
        print("Skipping Step 4: FAISS index already exists.")

    # --- Step 5: Load FAISS Index ---
    print("\nLoading FAISS index for matching...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if not os.path.exists(INDEX_SUMMARY_WITH_METADATA):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_SUMMARY_WITH_METADATA}. Please run the script again."
        )
    vector_store = FAISS.load_local(
        INDEX_SUMMARY_WITH_METADATA, embeddings, allow_dangerous_deserialization=True
    )

    # --- Step 6: Process Mentees ---
    print("\nProcessing mentees...")
    all_matches = []
    for mentee_subdir in os.listdir(mentee_dir):
        mentee_subdir_path = os.path.join(mentee_dir, mentee_subdir)
        if os.path.isdir(mentee_subdir_path):
            json_files = [
                f for f in os.listdir(mentee_subdir_path) if f.lower().endswith(".json")
            ]
            if not json_files:
                continue

            mentee_json_path = os.path.join(mentee_subdir_path, json_files[0])
            with open(mentee_json_path, "r") as f:
                mentee_data = json.load(f)

            mentee_preferences = mentee_data.get("research_Interest", [])
            cv_filename_base = mentee_data.get("submissions_files", [None])[0]
            if not cv_filename_base:
                continue

            mentee_cv_path = None
            for f in os.listdir(mentee_subdir_path):
                if f.endswith(cv_filename_base):
                    mentee_cv_path = os.path.join(mentee_subdir_path, f)
                    break

            if not mentee_cv_path:
                continue

            print(
                f"Processing {mentee_data.get('first_name')} {mentee_data.get('last_name')}..."
            )
            mentee_results = await process_single_mentee(
                mentee_cv_path=mentee_cv_path,
                vector_store=vector_store,
                mentee_preferences=mentee_preferences,
                mentee_data=mentee_data,
                k=num_mentors,
            )
            if mentee_results:
                all_matches.append(mentee_results)

    # --- Step 7: Save Final Output ---
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, "best_matches.json")
    with open(json_output_path, "w") as f:
        json.dump(all_matches, f, indent=4)
    print(f"\nAll mentee matches saved to {json_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mentor Matching Pipeline")
    parser.add_argument(
        "--mentees",
        required=True,
        help="Path to the directory containing mentee subdirectories.",
    )
    parser.add_argument(
        "--mentors",
        required=False,
        help="Path to the directory containing mentor resumes. Only required if mentor_data.csv doesn't exist or --overwrite is used.",
    )
    parser.add_argument(
        "--num_mentors",
        type=int,
        required=True,
        help="Number of desired matches for evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-processing of all mentor data from scratch.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the final JSON output. Defaults to 'output/' in the project root.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.mentees,
            args.mentors,
            args.num_mentors,
            args.overwrite,
            args.output_dir,
        )
    )
