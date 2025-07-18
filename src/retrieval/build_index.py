import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils import find_professor_type, rank_professors
from src.config import paths
from src.config.model import LLM_MODEL, EMBEDDING_MODEL


def build_index():
    """
    Builds and saves FAISS vector stores from mentor data.
    If the ranked data file already exists, it uses it. Otherwise, it creates it first.
    """
    load_dotenv()
    llm = ChatOpenAI(model=LLM_MODEL)

    if os.path.exists(paths.PATH_TO_MENTOR_DATA_RANKED):
        print(f"Loading existing ranked data from {paths.PATH_TO_MENTOR_DATA_RANKED}")
        merged_df = pd.read_csv(paths.PATH_TO_MENTOR_DATA_RANKED, sep="\t")
    else:
        print("Ranked data not found. Creating it from summaries...")
        summary_df = pd.read_csv(paths.PATH_TO_SUMMARY, sep="\t")

        # Add Professor_Type
        summary_df["Professor_Type"] = [
            find_professor_type(text) for text in summary_df["Mentor_Data"].fillna("")
        ]

        # Add Rank
        merged_df = rank_professors(summary_df)

        # Save the ranked data
        merged_df.to_csv(paths.PATH_TO_MENTOR_DATA_RANKED, sep="\t", index=False)
        print(f"Saved ranked mentor data to {paths.PATH_TO_MENTOR_DATA_RANKED}")

    # Ensure we have only the required columns
    merged_df = merged_df[
        ["Mentor_Data", "Mentor_Profile", "Mentor_Summary", "Professor_Type", "Rank"]
    ]

    # Create documents for assistant professors and above (Rank >= 1)
    docs_assistant_and_above = [
        p + "\n=====\n" + s
        for p, s, r in zip(
            merged_df["Mentor_Profile"].values,
            merged_df["Mentor_Summary"].values,
            merged_df["Rank"].values,
        )
        if r >= 1
    ]

    # Create documents for ranks higher than assistant professor (Rank > 1)
    docs_above_assistant = [
        p + "\n=====\n" + s
        for p, s, r in zip(
            merged_df["Mentor_Profile"].values,
            merged_df["Mentor_Summary"].values,
            merged_df["Rank"].values,
        )
        if r > 1
    ]

    docs_with_metadata = []
    # Create documents with metadata for both collections
    for _, row in merged_df.iterrows():
        # Create metadata dictionary
        doc_metadata = {
            "Mentor_Profile": row["Mentor_Profile"],
            "Professor_Type": row["Professor_Type"],
            "Rank": row["Rank"],
        }

        # Create document with page_content as Mentor_Summary and the metadata
        doc = Document(page_content=row["Mentor_Summary"], metadata=doc_metadata)
        # Append to the list
        docs_with_metadata.append(doc)

    # Create vector stores
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store_docs_with_metadata = FAISS.from_documents(
        documents=docs_with_metadata, embedding=embeddings
    )
    vector_store_assistant_and_above = FAISS.from_texts(
        texts=docs_assistant_and_above, embedding=embeddings
    )
    vector_store_above_assistant = FAISS.from_texts(
        texts=docs_above_assistant, embedding=embeddings
    )

    # Create retrievers
    retriever_docs_with_metadata = vector_store_docs_with_metadata.as_retriever()
    retriever_assistant_and_above = vector_store_assistant_and_above.as_retriever()
    retriever_above_assistant = vector_store_above_assistant.as_retriever()

    # Save vector stores
    vector_store_docs_with_metadata.save_local(paths.INDEX_SUMMARY_WITH_METADATA)
    vector_store_assistant_and_above.save_local(paths.INDEX_SUMMARY_ASSISTANT_AND_ABOVE)
    vector_store_above_assistant.save_local(paths.INDEX_SUMMARY_ABOVE_ASSISTANT)

    print("Vector stores created and saved successfully.")

    return (
        vector_store_assistant_and_above,
        retriever_assistant_and_above,
        vector_store_above_assistant,
        retriever_above_assistant,
        vector_store_docs_with_metadata,
        retriever_docs_with_metadata,
    )


if __name__ == "__main__":
    build_index()
