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
import base64

# Load .env file
load_dotenv()

from bin.build_index import main as build_index
from bin.mentor_mentee_data_generator_gpt4o import generate_mock_cv
from bin.search_candidate_mentors import search_candidate_mentors
from bin.evaluate_matches import evaluate_pair_with_llm, extract_eval_scores_with_llm, instructions
from bin.html_table_generator import create_mentor_table_html_and_csv_data
from bin.utils_ import clean_summary, extract_and_format_name

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
vector_store_docs_with_metadata = None
retriever_assistant_and_above = None
retriever_above_assistant = None
retriever_docs_with_metadata = None

def load_or_build_indices():
    global vector_store_assistant_and_above, vector_store_above_assistant, vector_store_docs_with_metadata
    global retriever_assistant_and_above, retriever_above_assistant, retriever_docs_with_metadata

    # Check if indices already exist
    if os.path.exists("db/index_summary_assistant_and_above") and os.path.exists("db/index_summary_above_assistant") and os.path.exists("db/index_summary_with_metadata"):
        from langchain_community.vectorstores import FAISS
        vector_store_assistant_and_above = FAISS.load_local("db/index_summary_assistant_and_above", embeddings, allow_dangerous_deserialization = True)
        vector_store_above_assistant = FAISS.load_local("db/index_summary_above_assistant", embeddings, allow_dangerous_deserialization = True)
        vector_store_docs_with_metadata = FAISS.load_local("db/index_summary_with_metadata", embeddings, allow_dangerous_deserialization = True)
    else:
        # Build indices if they don't exist
        vector_store_assistant_and_above, retriever_assistant_and_above, vector_store_above_assistant, retriever_above_assistant, vector_store_docs_with_metadata, retriever_docs_with_metadata = build_index()

    # Create retrievers if they don't exist
    if retriever_assistant_and_above is None:
        retriever_assistant_and_above = vector_store_assistant_and_above.as_retriever(search_kwargs = search_kwargs)
    if retriever_above_assistant is None:
        retriever_above_assistant = vector_store_above_assistant.as_retriever()
    if retriever_docs_with_metadata is None:
        retriever_docs_with_metadata = vector_store_docs_with_metadata.as_retriever()

def load_professor_types():
    global professor_types
    with open("./data/professor_types.txt", "r") as f:
        professor_types = f.read().splitlines()

# Load or build indices at startup
load_or_build_indices()
# Also load professor types from list created in build_index.py
load_professor_types()

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


async def process_cv_async(file, num_candidates, selected_professor_types=None):
    try:
        # Check file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension not in ['.pdf', '.docx']:
            raise ValueError(f"Unsupported file type: {file_extension}. Please use .pdf or .docx")

        mentee, pdf_text = await generate_mock_cv(file.name)
        print("Generated mock CV and extracted text")

        # Choose the appropriate vector store based on mentee's status
        vector_store = vector_store_assistant_and_above if not mentee.is_assistant_professor else vector_store_above_assistant

        # Set up metadata filter if professor types are selected
        metadata_filter = None
        if selected_professor_types and len(selected_professor_types) > 0:
            # Create a lambda function that checks if a document's professor_type is in the selected list
            metadata_filter = lambda metadata: metadata.get("professor_type") in selected_professor_types
            print(f"Filtering for professor types: {selected_professor_types}")

        # Pass the filter to the search function
        search_results = await search_candidate_mentors(
            k=num_candidates, 
            mentee_cv_text=pdf_text, 
            vector_store=vector_store,
            metadata_filter=metadata_filter
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

        # Generate CSV string
        csv_string = generate_csv_string(csv_data)

        return mentee_summary, mentor_table_html, evaluated_matches, csv_string

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

def process_cv_wrapper(file, num_candidates, selected_professor_types=None):
    async def async_wrapper():
        return await process_cv_async(file, num_candidates, selected_professor_types)
    return asyncio.run(async_wrapper())

# New function to handle chat queries with streaming
async def chat_query(message, history, selected_professor_types=None):
    # Use the retriever with metadata - no filtering
    retriever = retriever_docs_with_metadata
    
    # Get relevant documents without professor type filtering
    docs = retriever.get_relevant_documents(message)
    
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

    partial_message = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield "", history + [[message, partial_message]]  # Return "" for the input box
    # Prepare context from filtered documents
    context = "\n\n".join([doc.page_content for doc in filtered_docs])
    
    # Prepare the messages for the OpenAI model
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions about matching mentors and mentees."},
        {"role": "system", "content": "You are a helpful assistant answering questions about matching potential collaborators."},
        {"role": "user", "content": f"Based on the following context, answer the user's question:\n\nContext:\n{context}\n\nUser's question: {message}"}
    ]
    
    # Add conversation history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    # Generate response using OpenAI with streaming
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1.0,
        stream=True
    )

    partial_message = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield "", history + [[message, partial_message]]  # Return "" for the input box
    # Prepare context from filtered documents
    context = "\n\n".join([doc.page_content for doc in filtered_docs])
    
    # Prepare the messages for the OpenAI model
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions about matching mentors and mentees."},
        {"role": "system", "content": "You are a helpful assistant answering questions about matching potential collaborators."},
        {"role": "user", "content": f"Based on the following context, answer the user's question:\n\nContext:\n{context}\n\nUser's question: {message}"}
    ]
    
    # Add conversation history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    # Generate response using OpenAI with streaming
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1.0,
        stream=True
    )

    full_response = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            yield history + [[message, full_response]], ""

#def process_cv_wrapper(file, num_candidates):
#    loop = asyncio.get_event_loop()
#    return loop.run_until_complete(process_cv_async(file, num_candidates))

#def chat_query_wrapper(message, history, index_choice):
#    loop = asyncio.get_event_loop()
#    return loop.run_until_complete(chat_query(message, history, index_choice))

# Gradio interface
with gr.Blocks(css=css) as demo:
    gr.HTML("<h1>TCH Mentor-Mentee Matching System</h1>")
    
    with gr.Tab("Mentor Search"):
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Upload Mentee CV (PDF)")

            with gr.Column(scale=1):
                num_candidates = gr.Number(label="Number of Candidates", value=5, minimum=1, maximum=100, step=1)
                
                # Add professor type selection for Mentor Search
                mentor_professor_types = gr.CheckboxGroup(
                    choices=professor_types,
                    label="Filter by Professor Types",
                    value=[]  # Default to no selection, which will include all
                )
                
                submit_btn = gr.Button("Submit")

        summary = gr.Textbox(label="Student CV Summary")
        mentor_table = gr.HTML(label="Matching Mentors Table")
        evaluated_matches = gr.State([])
        csv_data = gr.State()
        download_link = gr.HTML(label="Download CSV")

        # Define click events INSIDE the Blocks context
        submit_btn.click(
            fn=process_cv_wrapper,
            inputs=[file, num_candidates, mentor_professor_types],
            outputs=[summary, mentor_table, evaluated_matches, csv_data],
            api_name='search_cv'
        ).then(
            fn=get_csv_download,
            inputs=[csv_data],
            outputs=[download_link]
        )
        
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Type your message here...")
        clear = gr.Button("Clear Chat")

        # Define chat events INSIDE the Blocks context
        msg.submit(chat_query, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ([], None), None, [chatbot, msg], queue=False)

# Only after defining everything within the Blocks context, launch the app
if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, show_error=True)
