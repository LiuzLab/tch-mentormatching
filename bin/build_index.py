import pandas as pd
import os
from dotenv import load_dotenv

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader


load_dotenv()
PATH_TO_SUMMARY = "./simulated_data/mentor_student_cvs_with_summaries.csv"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo-0125"  # will change it :)


def main():
    llm = ChatOpenAI(model=MODEL_NAME)
    loader = CSVLoader(file_path=PATH_TO_SUMMARY, source_column="Mentor_Summary")
    docs = loader.load()
    vector_store = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriver = vector_store.as_retriever()
    vector_store.save_local("db/index_summary")


if __name__ == "__main__":
    main()
