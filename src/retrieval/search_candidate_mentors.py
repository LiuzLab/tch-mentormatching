import os
import asyncio
from src.config.client import get_async_openai_client
from src.generate_text import generate_text_async
from src.config.prompts import mentee_instructions

client = None


async def search_candidate_mentors(
    k=36, mentee_cv_text="", vector_store=None, metadata_filter=None
):
    global client
    if client is None:
        client = get_async_openai_client()
    from langchain_community.vectorstores import FAISS

    if vector_store is None:
        raise ValueError("Vector store must be provided")

    print("Starting to generate mentee CV summary")
    mentee_cv_summary = await generate_text_async(
        client, mentee_cv_text, mentee_instructions
    )
    print("Finished generating mentee CV summary")

    print("Starting similarity search")

    if metadata_filter:
        print(f"Applying metadata filter for professor types first")

        # Step 1: Collect all documents matching the metadata filter
        all_filtered_docs = []
        filter_count = 0

        # Iterate through all documents in the vector store
        for doc_id, doc in vector_store.docstore._dict.items():
            if metadata_filter(doc.metadata):
                all_filtered_docs.append(doc)
                filter_count += 1

        print(f"Found {filter_count} documents matching professor type filter")

        if not all_filtered_docs:
            print("Warning: No documents match the metadata filter")
            return {"mentee_cv_summary": mentee_cv_summary, "candidates": []}

        # Step 2: Create a new vector store containing only the filtered documents
        # Handle different versions of LangChain which might use different attribute names
        try:
            # Try accessing the embeddings attribute (newer versions)
            embeddings = vector_store.embeddings
        except AttributeError:
            try:
                # Try the older _embedding_function attribute
                embeddings = vector_store._embedding_function
            except AttributeError:
                # Fall back to the global embeddings object if available
                if "embeddings" in globals():
                    embeddings = globals()["embeddings"]
                else:
                    raise ValueError("Could not find embeddings in vector store")

        # Add ids to documents if they don't have them, to avoid errors in similarity search
        from uuid import uuid4

        for i, doc in enumerate(all_filtered_docs):
            if not hasattr(doc, "id") or doc.id is None:
                # Try to set id as attribute
                try:
                    doc.id = str(uuid4())
                except AttributeError:
                    # If we can't set it as an attribute, create a new document with id
                    from langchain_core.documents import Document

                    # Copy all existing attributes
                    new_doc = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata,
                        id=str(uuid4()),
                    )
                    all_filtered_docs[i] = new_doc

        # Create new vector store with the prepared documents
        filtered_vector_store = FAISS.from_documents(all_filtered_docs, embeddings)

        # Step 3: Perform similarity search on the filtered vector store
        print(f"Performing similarity search on {filter_count} filtered documents")
        candidates_with_scores = filtered_vector_store.similarity_search_with_score(
            mentee_cv_summary, k=min(k, filter_count)
        )
    else:
        # If no filter is provided, use the original vector store
        candidates_with_scores = vector_store.similarity_search_with_score(
            mentee_cv_summary, k=k
        )

    print("Finished similarity search")

    # Debug logging
    if candidates_with_scores:
        print(f"Found {len(candidates_with_scores)} matching candidates")
        print("Debug: First candidate structure:")
        print(f"Metadata: {candidates_with_scores[0][0].metadata}")
        print(
            f"Page content preview: {candidates_with_scores[0][0].page_content[:100]}..."
        )
        print(f"Similarity score: {candidates_with_scores[0][1]}")
    else:
        print("Warning: No candidates found that match the criteria")

    return {
        "mentee_cv_summary": mentee_cv_summary,
        "candidates": candidates_with_scores,
    }


if __name__ == "__main__":
    # You can add test code here if needed
    async def main():
        # Example usage
        vector_store = FAISS.load_local("path_to_your_faiss_index", OpenAIEmbeddings())

        # Example of using with a metadata filter
        def example_filter(metadata):
            return metadata.get("professor_type") in [
                "Assistant Professor",
                "Associate Professor",
            ]

        result = await search_candidate_mentors(
            k=5,
            mentee_cv_text="Sample CV text",
            vector_store=vector_store,
            metadata_filter=example_filter,
        )

        print(result["mentee_cv_summary"])
        print(len(result["candidates"]))

    asyncio.run(main())
