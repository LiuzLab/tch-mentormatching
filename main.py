import os
import pandas as pd
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
# Load .env file
load_dotenv()

from bin.build_index import main as build_index
from bin.mentor_mentee_data_generator_gpt4o import generate_mock_cv
from bin.search_candidate_mentors import search_candidate_mentors
from bin.evaluate_matches import evaluate_pair_with_llm, extract_scores, instructions
from bin.correct_csv_columns import correct_csv_columns
from bin.html_table_generator import create_mentor_table_html


# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

# Build index (this might take some time, consider doing this separately)
vector_store, retriever = build_index()

# note 5 candidates takes short time; 36 candidates seems to take 10-15 minutes
def process_cv(file, num_candidates):
    try:
        # Generate mock CV and extract PDF text
        mock_cv, pdf_text = generate_mock_cv(file.name)
        print("Generated mock CV and extracted PDF text")
        
        # Search for candidate mentors
        search_results = search_candidate_mentors(k=num_candidates, 
                                                  mentee_cv_text=pdf_text, 
                                                  vector_store=vector_store)
        mentee_summary = search_results["mentee_cv_summary"]
        candidates = search_results["candidates"]
        
        # Evaluate and rank matches
        evaluated_matches = []
        for candidate, similarity_score in candidates:
            match_res = evaluate_pair_with_llm(
                client, 
                candidate.page_content, 
                mentee_summary, 
                instructions
            )
            criterion_scores = extract_scores(match_res)
            evaluated_matches.append({
                'Mentor Summary': candidate.page_content,
                'Similarity Score': f"{similarity_score:.4f}",
                'Evaluation': match_res,
                'Criterion Scores': criterion_scores
            })

        
        # Sort matches by overall match quality score
        evaluated_matches.sort(key=lambda x: x['Criterion Scores']['Overall Match Quality'] or 0, reverse=True)
        
        # Create custom HTML table
        mentor_table_html = create_mentor_table_html(evaluated_matches)
        
        return mentee_summary, "Matching Mentors", mentor_table_html
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred", "Error occurred", "Error occurred"

iface = gr.Interface(
    fn=process_cv,
    inputs=[gr.File(label="Upload Mentee CV (PDF)"),
        gr.Number(label="Number of Candidates", value=5, minimum=1, maximum=50, step=1)]
    ,
    outputs=[
        gr.Textbox(label="Student CV Summary"),
        gr.Textbox(label="Matching Mentors"),
        gr.HTML(label="Matching Mentors Table")

    ],
    title="TCH Mentor-Mentee Matching System",
    description="Please Upload a mentee's CV."
)

if __name__ == "__main__":
    iface.launch(share = True)