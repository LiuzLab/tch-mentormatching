import pandas as pd
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader


load_dotenv()
PATH_TO_SUMMARY = "./data/mentor_data_with_summaries.csv"
# Currently we are reading in this mentor data file and merging below b/c dont want to pay $250 
# to run batch_summarize_pdfs.py again. Just using output from first run and merging
PATH_TO_MENTOR_DATA = "./data/mentor_data.csv"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo-0125"  # will change it :)


def main():
    llm = ChatOpenAI(model=MODEL_NAME)
    summary_df = pd.read_csv(PATH_TO_SUMMARY, sep="\t")
    mentor_data_df = pd.read_csv(PATH_TO_MENTOR_DATA)

    # Merge dataframes on Mentor_Data column
    merged_df = summary_df.merge(mentor_data_df, on="Mentor_Data", how="left")

    # Ensure we have only the required columns after merging
    merged_df = merged_df[["Mentor_Data", "Mentor_Profile", "Mentor_Summary"]]

    docs = [
        p + "\n=====\n" + s
        for p, s in zip(merged_df["Mentor_Profile"].values, merged_df["Mentor_Summary"].values)
    ]

    vector_store = FAISS.from_texts(texts=docs, embedding=OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    vector_store.save_local("db/index_summary")
    return vector_store, retriever


if __name__ == "__main__":
    main()
