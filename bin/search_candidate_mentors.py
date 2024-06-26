import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from generate_text import generate_text
import batch_summarize_pdfs 
from openai import OpenAI
import evaluate_matches
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-3.5-turbo-0125"  # will change it :) 
client = OpenAI(api_key=OPENAI_KEY)

def search_candidate_mentors(k = 36, mentee_cv_text = ""):

    db = FAISS.load_local("./db/index_summary/", 
                          OpenAIEmbeddings(), 
                          allow_dangerous_deserialization=True)
    mentee_cv_summary = generate_text(client, mentee_cv_text, 
                                      batch_summarize_pdfs.mentee_instructions)
    candidates = db.similarity_search_with_score(mentee_cv_summary, k=k, fetch_k=k) 

    return {
        "mentee_cv_summary": mentee_cv_summary,
        "candidates": candidates

    }


if __name__ == "__main__":
    summary = """Johnathan A. Doe is a graduate of Houston University where he obtained his Bachelor's Science in Biology. He served as a Research Assistant in the Department of Pathology of the same university, where he displayed exceptional expertise in molecular biology and microbiota studies. Particularly, Doe has made significant contributions in microbial culture research.
                During his tenure, Doe has conducted experiments on bacterial culture pH readouts using UV-Vis absorption spectrophotometry, showcasing his skills in Molecular & Cellular Biology Techniques as well as data analysis using statistical software like R and SPSS. His research has resulted in notable publications such as "Analyzing Microbial Culture pH through UV-Vis Absorption Spectrophotometry," "Effectiveness of Molecular Models in Understanding Protein-Ligand Interactions," and "Investigating the Role of Microbiota on Human Immune Responses." 
                More than his research profile, Doe is actively involved in community health activities, volunteering in Houston Community Health Clinic and coordinating bioresearch events in his university. His diverse skillset, community involvement, and noteworthy publishing record highlight his suitability for collaboration or mentorship."""
    res = search_candidate_mentors(
        k=3,
        mentee_cv_text=summary
    )

    for candidate in res['candidates']:
        match_res = evaluate_matches.evaluate_pair_with_llm(client, 
                                                            candidate, 
                                                            summary, 
                                                            evaluate_matches.instructions)
        print(match_res)