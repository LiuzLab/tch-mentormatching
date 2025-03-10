import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .generate_text import generate_text_async
from .batch_summarize_pdfs import mentee_instructions, initialize_async_openai_client

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo-0125"

# Initialize AsyncOpenAI client
#client = initialize_async_openai_client()

from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def search_candidate_mentors(k=36, mentee_cv_text="", vector_store=None, metadata_filter=None):
    if vector_store is None:
        raise ValueError("Vector store must be provided")
    
    print("Starting to generate mentee CV summary")
    mentee_cv_summary = await generate_text_async(client, mentee_cv_text, mentee_instructions)
    print("Finished generating mentee CV summary")
    
    print("Starting similarity search")
    # Apply metadata filter if provided
    if metadata_filter:
        print(f"Applying metadata filter for professor types")
        candidates_with_scores = vector_store.similarity_search_with_score(
            mentee_cv_summary, k=k, fetch_k=k*2, filter=metadata_filter
        )
    else:
        candidates_with_scores = vector_store.similarity_search_with_score(
            mentee_cv_summary, k=k, fetch_k=k
        )
    print("Finished similarity search")
    
    # Debug logging
    if candidates_with_scores:
        print("Debug: First candidate structure:")
        print(f"Metadata: {candidates_with_scores[0][0].metadata}")
        print(f"Page content preview: {candidates_with_scores[0][0].page_content[:100]}...")
    else:
        print("Warning: No candidates found that match the criteria")

    return {"mentee_cv_summary": mentee_cv_summary, "candidates": candidates_with_scores}


if __name__ == "__main__":
    # You can add test code here if needed
    async def main():
        # Example usage
        vector_store = FAISS.load_local("path_to_your_faiss_index", OpenAIEmbeddings())
        
        # Example of using with a metadata filter
        def example_filter(metadata):
            return metadata.get("professor_type") in ["Assistant Professor", "Associate Professor"]
            
        result = await search_candidate_mentors(
            k=5, 
            mentee_cv_text="Sample CV text", 
            vector_store=vector_store,
            metadata_filter=example_filter
        )
        
        print(result["mentee_cv_summary"])
        print(len(result["candidates"]))

    asyncio.run(main())