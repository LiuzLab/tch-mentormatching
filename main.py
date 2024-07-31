import os
import re
import pandas as pd
from io import StringIO
from openai import OpenAI, AsyncOpenAI
import gradio as gr
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import traceback
import tempfile

# Load .env file
load_dotenv()

from bin.build_index import main as build_index
from bin.mentor_mentee_data_generator_gpt4o import generate_mock_cv
from bin.search_candidate_mentors import search_candidate_mentors
from bin.evaluate_matches import evaluate_pair_with_llm, extract_eval_scores_with_llm, instructions
from bin.correct_csv_columns import correct_csv_columns
from bin.html_table_generator import create_mentor_table_html_and_csv_data
from bin.utils import clean_summary, extract_and_format_name 

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
#client = OpenAI(api_key=OPENAI_KEY)
client = AsyncOpenAI(api_key=OPENAI_KEY) #asynchronous 

# Build index (this might take some time, consider doing this separately)
vector_store, retriever = build_index()


async def evaluate_match(client, candidate_tuple, mentee_summary):
    candidate, similarity_score = candidate_tuple
    # Assuming the mentor_id is at the beginning of the page_content
    mentor_id = candidate.page_content.split("===")[0].strip().replace(".txt", "")
    
    match_res = await evaluate_pair_with_llm(client, candidate.page_content, mentee_summary, instructions)
    criterion_scores = await extract_eval_scores_with_llm(client, match_res)
    return {
        "Mentor Summary": candidate.page_content,
        "Similarity Score": f"{similarity_score:.4f}",
        "Evaluation": match_res,
        "Criterion Scores": criterion_scores,
        "mentor_id": mentor_id,
    }

async def process_cv_async(file, num_candidates):
    try:
        # Generate mock CV and extract PDF text
        mock_cv, pdf_text = await generate_mock_cv(file.name)
        print("Generated mock CV and extracted PDF text")

        # Search for candidate mentors
        search_results = await search_candidate_mentors(
            k=num_candidates, mentee_cv_text=pdf_text, vector_store=vector_store
        )
        mentee_summary = search_results["mentee_cv_summary"]
        candidates = search_results["candidates"]

        # Evaluate and rank matches asynchronously
        tasks = [evaluate_match(client, candidate_tuple, mentee_summary) for candidate_tuple in candidates]
        evaluated_matches = await asyncio.gather(*tasks)

        # Sort matches by overall match quality score
        evaluated_matches.sort(
            key=lambda x: x["Criterion Scores"]["Overall Match Quality"] or 0,
            reverse=True,
        )

        # Create custom HTML table
        mentor_table_html, csv_data = create_mentor_table_html_and_csv_data(evaluated_matches)

        return mentee_summary, mentor_table_html, evaluated_matches, csv_data
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        import traceback
        print(traceback.format_exc())
        return "Error occurred", "Error occurred", []


def download_csv(csv_data):
    if not csv_data:
        return None
    
    df = pd.DataFrame(csv_data)
    
    # Create a temporary file with a fixed name
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
        df.to_csv(temp_file.name, index=False)
        temp_file_path = temp_file.name
    
    return temp_file_path



# Function to read CSS files
def read_css_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, 'static', 'css', filename)
    with open(css_path, 'r') as f:
        return f.read()

# Read CSS files
main_css = read_css_file('main.css')
mentor_table_css = read_css_file('mentor_table_styles.css')

# Combine CSS
css = main_css + mentor_table_css

def process_cv_wrapper(file, num_candidates):
    async def async_wrapper():
        return await process_cv_async(file, num_candidates)
    return asyncio.run(async_wrapper())


with gr.Blocks() as demo:
    gr.HTML("<h1>TCH Mentor-Mentee Matching System</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(label="Upload Mentee CV (PDF)")
        
        with gr.Column(scale=1):
            num_candidates = gr.Number(label="Number of Candidates", value=5, minimum=1, maximum=100, step=1)
            submit_btn = gr.Button("Submit")

    summary = gr.Textbox(label="Student CV Summary")

    mentor_table = gr.HTML(label="Matching Mentors Table", value="<div style='height: 500px;'>Results will appear here after submission.</div>")

    download_btn = gr.Button("Download Results as CSV")
    
    evaluated_matches = gr.State([])
    csv_data = gr.State([])

    submit_btn.click(
        fn=process_cv_wrapper,
        inputs=[file, num_candidates],
        outputs=[summary, mentor_table, evaluated_matches, csv_data],
        show_progress= True
    )
    
    download_btn.click(
        fn=download_csv,
        inputs=[csv_data],
        outputs=gr.File(label="Download CSV", height = 30),
        show_progress=False,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
