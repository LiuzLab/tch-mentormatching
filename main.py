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
from bin import evaluate_matches


# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

# Build index (this might take some time, consider doing this separately)
vector_store, retriever = build_index()

# For demonstration, we'll just return a message about the index
#index_info = "Vector index built and saved."
    

def process_cv(file):
    try:
        # Generate mock CV and extract PDF text
        mock_cv, pdf_text = generate_mock_cv(file.name)
        print("Generated mock CV and extracted PDF text")
        
        # Search for candidate mentors
        search_results = search_candidate_mentors(k=5, mentee_cv_text=pdf_text, vector_store=vector_store)
        mentee_summary = search_results["mentee_cv_summary"]
        candidates = search_results["candidates"]
        
        # Evaluate and rank matches
        evaluated_matches = []
        for candidate, score in candidates:
            match_res = evaluate_matches.evaluate_pair_with_llm(
                client, 
                candidate.page_content, 
                mentee_summary, 
                evaluate_matches.instructions
            )
            evaluated_matches.append({
                'Mentor Summary': candidate.page_content[:200] + "...",
                'Similarity Score': f"{score:.4f}",
                'Evaluation': match_res
            })
        
        # Sort matches by similarity score
        evaluated_matches.sort(key=lambda x: float(x['Similarity Score']), reverse=True)
        
        # Format the matching mentors as a table
        matching_mentors_df = pd.DataFrame(evaluated_matches)
        mentor_table = matching_mentors_df.to_html(index=False, escape=False)
        
        return mock_cv, mentee_summary, "Matching Mentors", mentor_table
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred", "Error occurred", "Error occurred", "Error occurred"

iface = gr.Interface(
    fn=process_cv,
    inputs=gr.File(label="Upload Mentee CV (PDF)"),
    outputs=[
        gr.Textbox(label="Generated Student CV"),
        gr.Textbox(label="Student CV Summary"),
        gr.Textbox(label="Matching Mentors"),
        gr.HTML(label="Matching Mentors Table")

    ],
    title="Student-Mentor Matching System",
    description="Upload a student's CV to generate a mock CV, summarize it, and find matching mentors."
)

if __name__ == "__main__":
    iface.launch(share = True) # remove reload if don't need to change as 