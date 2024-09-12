import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from .utils import find_professor_type, rank_professors

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"
PATH_TO_SUMMARY = "./data/mentor_data_with_summaries.csv"
PATH_TO_MENTOR_DATA = "./data/mentor_data.csv"
PATH_TO_SUMMARY_DATA = "./data/summary_data.csv"
PATH_TO_MENTOR_DATA_RANKED = "./data/mentor_data_summaries_ranks.csv"

# define the search kwargs for langchain FAISS retriever
search_kwargs={'k': 15, 'fetch_k': 50} #k is number of docs to return; fetch_k is number to search

def main():
    llm = ChatOpenAI(model=MODEL_NAME)

    # Check if ranked data exists
    if os.path.exists(PATH_TO_MENTOR_DATA_RANKED):
        print("Loading existing ranked data...")
        merged_df = pd.read_csv(PATH_TO_MENTOR_DATA_RANKED, sep="\t")
    else:
        print("Ranked data not found. Creating from existing or new data...")
        # Read the data
        summary_df = pd.read_csv(PATH_TO_SUMMARY, sep="\t")
        mentor_data_df = pd.read_csv(PATH_TO_MENTOR_DATA)
        
        # Merge dataframes on Mentor_Data column
        merged_df = summary_df.merge(mentor_data_df, on="Mentor_Data", how="left")
        
        # Add Professor_Type
        merged_df['Professor_Type'] = merged_df['Mentor_Data'].apply(find_professor_type)
        
        # Add Rank
        merged_df = rank_professors(merged_df)
        
        print(merged_df.head())
        
        # Save the ranked data
        merged_df.to_csv(PATH_TO_MENTOR_DATA_RANKED, sep="\t", index=False)
        print(f"Saved ranked mentor data to {PATH_TO_MENTOR_DATA_RANKED}")

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

    # Create vector stores
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",# trying newest embedding model
                                      dimensions=3072  # Request full dimensionality
                                ) #specify dimensions bc of mismatch error during chat
    # debug check dimensions
    test_vector = embeddings.embed_query("test")
    print(f"Test vector shape: {len(test_vector)}")
    assert len(test_vector) == 3072, f"Expected vector dimension 3072, but got {len(test_vector)}"
    

    vector_store_assistant_and_above = FAISS.from_texts(texts=docs_assistant_and_above, embedding=embeddings)
    vector_store_above_assistant = FAISS.from_texts(texts=docs_above_assistant, embedding=embeddings)

    # Create retrievers
    retriever_assistant_and_above = vector_store_assistant_and_above.as_retriever(search_kwargs = search_kwargs) # try increasing k
    retriever_above_assistant = vector_store_above_assistant.as_retriever(search_kwargs = search_kwargs) # try increasing k

    # After creating the index, verify its dimension
    print(f"FAISS index dimension: {vector_store_assistant_and_above.index.d}")
    assert vector_store_assistant_and_above.index.d == 3072, f"Expected index dimension 3072, but got {vector_store_assistant_and_above.index.d}"
    
    # Save vector stores
    vector_store_assistant_and_above.save_local("db/index_summary_assistant_and_above")
    vector_store_above_assistant.save_local("db/index_summary_above_assistant")

    print("Vector stores created and saved successfully.")

    return (vector_store_assistant_and_above, retriever_assistant_and_above, 
            vector_store_above_assistant, retriever_above_assistant)

if __name__ == "__main__":
    main()