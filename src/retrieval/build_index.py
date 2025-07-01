import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils import find_professor_type, rank_professors
from src.config import paths
from src.config.model import LLM_MODEL

def main(df=None):
    load_dotenv()
    llm = ChatOpenAI(model=LLM_MODEL)

    if df is None:
        # Check if ranked data exists
        if os.path.exists(paths.PATH_TO_MENTOR_DATA_RANKED):
            print("Loading existing ranked data...")
            merged_df = pd.read_csv(paths.PATH_TO_MENTOR_DATA_RANKED, sep="\t")
        else:
            print("Ranked data not found. Creating from existing or new data...")
            # Read the data
            summary_df = pd.read_csv(paths.PATH_TO_SUMMARY, sep="\t")
            mentor_data_df = pd.read_csv(paths.PATH_TO_MENTOR_DATA)
            
            # Merge dataframes on Mentor_Data column
            merged_df = summary_df.merge(mentor_data_df, on="Mentor_Data", how="left")
            
            # Add Professor_Type
            merged_df['Professor_Type'] = merged_df['Mentor_Data'].apply(find_professor_type)
            
            # Add Rank
            merged_df = rank_professors(merged_df)
            
            print(merged_df.head())
            
            # Save the ranked data
            merged_df.to_csv(paths.PATH_TO_MENTOR_DATA_RANKED, sep="\t", index=False)
            print(f"Saved ranked mentor data to {paths.PATH_TO_MENTOR_DATA_RANKED}")
    else:
        merged_df = df

    # Save the unique Professor Types to a file
    unique_professor_types = merged_df["Professor_Type"].unique()
    with open(paths.PROFESSOR_TYPES_PATH, "w") as f:
        for pt in unique_professor_types:
            f.write(pt + "\n")
    print(f"Saved unique Professor Types to {paths.PROFESSOR_TYPES_PATH}")


    # Ensure we have only the required columns
    merged_df = merged_df[["Mentor_Data", "Mentor_Profile", "Mentor_Summary", "Professor_Type", "Rank"]]

    # Create documents for assistant professors and above (Rank >= 1)
    docs_assistant_and_above = [
        p + "\n=====\n" + s
        for p, s, r in zip(merged_df["Mentor_Profile"].values, merged_df["Mentor_Summary"].values, merged_df["Rank"].values)
        if r >= 1
    ]

    # Create documents for ranks higher than assistant professor (Rank > 1)
    docs_above_assistant = [
        p + "\n=====\n" + s
        for p, s, r in zip(merged_df["Mentor_Profile"].values, merged_df["Mentor_Summary"].values, merged_df["Rank"].values)
        if r > 1
    ]

    docs_with_metadata = []
    # Create documents with metadata for both collections
    for _, row in merged_df.iterrows():
        # Create metadata dictionary
        doc_metadata = {
            "Mentor_Profile": row["Mentor_Profile"],
            "Professor_Type": row["Professor_Type"],
            "Rank": row["Rank"]
        }
        
        # Create document with page_content as Mentor_Summary and the metadata
        doc = Document(page_content=row["Mentor_Summary"], metadata=doc_metadata)
        # Append to the list
        docs_with_metadata.append(doc)

    # Create vector stores
    embeddings = OpenAIEmbeddings()
    vector_store_docs_with_metadata = FAISS.from_documents(documents=docs_with_metadata, embedding=embeddings)
    vector_store_assistant_and_above = FAISS.from_texts(texts=docs_assistant_and_above, embedding=embeddings)
    vector_store_above_assistant = FAISS.from_texts(texts=docs_above_assistant, embedding=embeddings)

    # Create retrievers
    retriever_docs_with_metadata = vector_store_docs_with_metadata.as_retriever()
    retriever_assistant_and_above = vector_store_assistant_and_above.as_retriever()
    retriever_above_assistant = vector_store_above_assistant.as_retriever()

    # Save vector stores
    vector_store_docs_with_metadata.save_local(paths.INDEX_SUMMARY_WITH_METADATA)
    vector_store_assistant_and_above.save_local(paths.INDEX_SUMMARY_ASSISTANT_AND_ABOVE)
    vector_store_above_assistant.save_local(paths.INDEX_SUMMARY_ABOVE_ASSISTANT)

    print("Vector stores created and saved successfully.")

    return (vector_store_assistant_and_above, retriever_assistant_and_above, 
            vector_store_above_assistant, retriever_above_assistant,
            vector_store_docs_with_metadata, retriever_docs_with_metadata)

if __name__ == "__main__":
    main()