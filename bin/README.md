# Mentor Matching Pipeline

## High-Level Overview

This project is a command-line application that automates the mentor-matching process. It operates in two main phases:

1.  **Ingestion & Indexing:** It processes a collection of mentor resumes, summarizes their content using an LLM, and builds a searchable vector database (a knowledge base) of the mentor profiles.
2.  **Matching & Evaluation (RAG):** It takes a mentee's CV, uses the vector database to **Retrieve** the most relevant mentors, and then uses an LLM to **Generate** a detailed evaluation and a match score for each candidate pair.

The entire process is orchestrated by the `bin/main.py` script, which acts as the central entry point.

---

### Key Modules and Their Roles

*   `bin/config/`: This directory centralizes all configuration, including API keys (via `.env`), file paths, model names, and LLM prompts. This makes the application easier to configure and maintain.
*   `bin/processing/`: This module handles data extraction and transformation. It reads raw resume files, converts them to text, and uses the OpenAI Batch API to efficiently summarize them.
*   `bin/retrieval/`: This is the core of the retrieval step. It builds the FAISS vector index from the mentor summaries and searches this index to find the best candidate mentors for a given mentee.
*   `bin/eval/`: This module handles the evaluation and presentation of the results. It uses an LLM to score the retrieved mentor-mentee pairs and generates a final HTML report.
*   `bin/utils.py`: Contains helper functions used across different modules, such as formatting names and ranking professors.

---

### Step-by-Step Pipeline Functionality

When you run `python main.py`, here is what happens in order:

**Phase 1: Mentor Data Processing**

1.  **Resume Ingestion (`main.py` -> `processing/io_utils.py`):**
    *   The pipeline starts by looking at the directory of mentor resumes you provide.
    *   It reads all supported files (`.pdf`, `.docx`, `.txt`), extracts the raw text from each, and creates a single CSV file: `data/mentor_data.csv`.

2.  **Batch Summarization (`main.py` -> `processing/batch.py`):**
    *   The script then takes `data/mentor_data.csv` and prepares a batch job for the OpenAI API.
    *   It sends all the mentor profiles to be summarized according to the `mentor_instructions` in `config/prompts.py`.
    *   The results are saved to a new file: `data/mentor_data_with_summaries.csv`.

3.  **Vector Indexing (`main.py` -> `retrieval/build_index.py`):**
    *   The script reads the CSV file containing the mentor summaries.
    *   It enriches the data by adding professor ranks and types using functions from `utils.py`.
    *   It converts the mentor summaries into numerical representations (embeddings) and stores them in a FAISS vector store located in the `db/` directory. This vector store is your indexed knowledge base, optimized for fast similarity searches.

**Phase 2: Mentee Matching (RAG)**

4.  **Mentee CV Summarization (`main.py` -> `retrieval/search_candidate_mentors.py`):**
    *   The script reads the mentee's CV file that you provide.
    *   It generates a concise summary of the mentee's profile using the `mentee_instructions` prompt.

5.  **Candidate Retrieval (`main.py` -> `retrieval/search_candidate_mentors.py`):**
    *   This is the **"Retrieval"** step. The mentee's summary is used to search the FAISS vector store.
    *   The system retrieves the top `k` (default is 10) mentors whose summaries are most similar to the mentee's summary.

6.  **LLM-Powered Evaluation (`main.py` -> `eval/evaluate_matches.py`):**
    *   This is the **"Generation"** step. For each retrieved mentor, the script pairs them with the mentee.
    *   It then asks the LLM to perform a detailed evaluation of the pair, providing a qualitative summary and quantitative scores for research interest, skillset, and availability.

7.  **Report Generation (`main.py` -> `eval/html_table_generator.py`):**
    *   The final evaluated matches are formatted into a user-friendly HTML table.
    *   The same data is also saved as a structured CSV file.

---

### How to Run the Pipeline

You can now run the entire process with a single command from within the `bin` directory.

**Inputs:**
*   `--mentee`: The file path to the mentee's CV.
*   `--mentors`: The path to the directory containing all the mentor resume files.

**Command:**
```bash
python main.py --mentee /path/to/your/mentee_cv.txt --mentors /path/to/the/mentor_resumes/
```

**Outputs:**
*   `mentor_matches.html`: A styled HTML table with the top matches, their summaries, and evaluation scores.
*   `mentor_matches.csv`: A CSV file with the same structured results for further analysis.
*   Intermediate files in the `data/` and `db/` directories that are used by the pipeline.
