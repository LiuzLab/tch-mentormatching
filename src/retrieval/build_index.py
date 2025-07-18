import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import paths
from src.config.model import EMBEDDING_MODEL


def build_index(df: pd.DataFrame):
    """
    Builds and saves a FAISS vector store from the mentor DataFrame.
    """
    # Ensure required columns are present
    required_cols = ["Mentor_Summary", "Mentor_Profile", "Professor_Type", "Rank"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"DataFrame must contain the following columns: {required_cols}"
        )

    # Create documents with metadata
    docs_with_metadata = []
    for _, row in df.iterrows():
        doc_metadata = {
            "Mentor_Profile": row["Mentor_Profile"],
            "Professor_Type": row["Professor_Type"],
            "Rank": row["Rank"],
        }
        doc = Document(page_content=row["Mentor_Summary"], metadata=doc_metadata)
        docs_with_metadata.append(doc)

    # Create and save the vector store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(
        documents=docs_with_metadata, embedding=embeddings
    )
    vector_store.save_local(paths.INDEX_SUMMARY_WITH_METADATA)

    print("Vector store created and saved successfully.")
