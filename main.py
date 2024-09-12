import os
import re
import pandas as pd
import numpy as np
from io import StringIO
from openai import OpenAI, AsyncOpenAI
import gradio as gr
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import traceback
import tempfile
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load .env file
load_dotenv()

from bin.build_index import main as build_index
from bin.mentor_mentee_data_generator_gpt4o import generate_mock_cv
from bin.search_candidate_mentors import search_candidate_mentors
from bin.evaluate_matches import evaluate_pair_with_llm, extract_eval_scores_with_llm, instructions
from bin.html_table_generator import create_mentor_table_html_and_csv_data
from bin.utils import clean_summary, extract_and_format_name 

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_KEY)
embeddings = OpenAIEmbeddings()

# define search kwargs
search_kwargs = {'k': 20, 'fetch_k': 100}  # Increasing the number of documents retrieved; #k is number of docs to return; fetch_k is number to search; default 4 and 20 respectively

# Global variables to store vector stores and retrievers
vector_store_assistant_and_above = None
vector_store_above_assistant = None
retriever_assistant_and_above = None
retriever_above_assistant = None

def load_or_build_indices():
    global vector_store_assistant_and_above, vector_store_above_assistant
    global retriever_assistant_and_above, retriever_above_assistant

    # Check if indices already exist
    if os.path.exists("db/index_summary_assistant_and_above") and os.path.exists("db/index_summary_above_assistant"):
        from langchain_community.vectorstores import FAISS
        vector_store_assistant_and_above = FAISS.load_local("db/index_summary_assistant_and_above", embeddings, allow_dangerous_deserialization = True)
        vector_store_above_assistant = FAISS.load_local("db/index_summary_above_assistant", embeddings, allow_dangerous_deserialization = True)
    else:
        # Build indices if they don't exist
        vector_store_assistant_and_above, retriever_assistant_and_above, vector_store_above_assistant, retriever_above_assistant = build_index()

    # Verify the loaded index dimensions
    print(f"Loaded index dimension: {vector_store_assistant_and_above.index.d}")
    assert vector_store_assistant_and_above.index.d == 3072, f"Expected index dimension 3072, but got {vector_store_assistant_and_above.index.d}"

    # Create retrievers if they don't exist
    if retriever_assistant_and_above is None:
        retriever_assistant_and_above = vector_store_assistant_and_above.as_retriever(search_kwargs = search_kwargs)
    if retriever_above_assistant is None:
        retriever_above_assistant = vector_store_above_assistant.as_retriever(search_kwargs = search_kwargs)

# Load or build indices at startup
load_or_build_indices()

async def evaluate_match(client, candidate_tuple, mentee_summary):
    candidate, similarity_score = candidate_tuple
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
        # Check file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension not in ['.pdf', '.docx']:
            raise ValueError(f"Unsupported file type: {file_extension}. Please use .pdf or .docx")

        mentee, pdf_text = await generate_mock_cv(file.name)
        print("Generated mock CV and extracted text")

        # Choose the appropriate vector store based on index_choice
        index_choice = "Assistant Professors and Above" if not mentee.is_assistant_professor else "Above Assistant Professors"
        vector_store = vector_store_assistant_and_above if index_choice == "Assistant Professors and Above" else vector_store_above_assistant

        search_results = await search_candidate_mentors(
            k=num_candidates, mentee_cv_text=pdf_text, vector_store=vector_store
        )
        mentee_summary = search_results["mentee_cv_summary"]
        candidates = search_results["candidates"]

        tasks = [evaluate_match(client, candidate_tuple, mentee_summary) for candidate_tuple in candidates]
        evaluated_matches = await asyncio.gather(*tasks)

        evaluated_matches.sort(
            key=lambda x: x["Criterion Scores"]["Overall Match Quality"] or 0,
            reverse=True,
        )

        mentor_table_html, csv_data = create_mentor_table_html_and_csv_data(evaluated_matches)

        return mentee_summary, mentor_table_html, evaluated_matches, csv_data
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return "Error occurred", "Error occurred", [], []

def download_csv(csv_data):
    if not csv_data:
        return None
    
    df = pd.DataFrame(csv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
        df.to_csv(temp_file.name, index=False)
        temp_file_path = temp_file.name
    
    return temp_file_path

def read_css_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, 'static', 'css', filename)
    with open(css_path, 'r') as f:
        return f.read()

main_css = read_css_file('main.css')
mentor_table_css = read_css_file('mentor_table_styles.css')
css = main_css + mentor_table_css

def process_cv_wrapper(file, num_candidates):
    async def async_wrapper():
        return await process_cv_async(file, num_candidates)
    return asyncio.run(async_wrapper())

# New function to handle chat queries with streaming
async def chat_query(message, history, index_choice):
    # Choose the appropriate retriever based on index_choice
    retriever = retriever_assistant_and_above if index_choice == "Assistant Professors and Above" else retriever_above_assistant
    
    # Use the retriever to get relevant documents
    docs = await retriever.ainvoke(message)
    
    # Prepare context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Prepare the messages for the OpenAI model
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions about matching mentors with mentees and matching potential collaborators. Focus on the research content and avoid mentioning personal information about researchers. Answer based only on the provided context."},
        {"role": "user", "content": f"Based solely on the following context, answer the user's question. If the information is not in the context, say you don't have enough information:\n\nContext:\n{context}\n\nUser's question: {message}"}
    ]
    
    # Add relevant history
    for past_message, past_response in history[-8:]:  # Include last 8 exchanges for context; might need to tune this based off API limits ahd best performances
        messages.append({"role": "user", "content": past_message})
        messages.append({"role": "assistant", "content": past_response})
    
    # Add the current query and context
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nCurrent question: {message}\n\nAnswer the current question based on the provided context. If the information isn't in the context, say you don't have enough information."})
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
        stream=True
    )

    full_response = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            yield history + [[message, full_response]], ""


# Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1>TCH Mentor-Mentee Matching System</h1>")
    
    with gr.Tab("Mentor Search"):
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
            show_progress=True
        )

        download_btn.click(
            fn=download_csv,
            inputs=[csv_data],
            outputs=gr.File(label="Download CSV", height=30),
            show_progress=False,
        )

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Type your message here...")
        clear = gr.Button("Clear Chat")

        chat_index_choice = gr.Dropdown(
            choices=["Assistant Professors and Above", "Above Assistant Professors"],
            label="Select Index for Chat",
            value="Assistant Professors and Above"
        )

        msg.submit(chat_query, inputs=[msg, chatbot, chat_index_choice], outputs=[chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)