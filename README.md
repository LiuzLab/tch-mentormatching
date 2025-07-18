# TCH Mentor-Matching

This project is a comprehensive pipeline designed to match mentees with suitable mentors based on their professional profiles and research interests, primarily using PDF resumes. It leverages Large Language Models (LLMs) for summarization, evaluation, and vector embeddings to find the best possible matches.

## Dataflow Diagrams

### Building Database

```mermaid
  flowchart LR;
     mentor_cv[/"Mentor CVs (PDFs)"/] --> text_extraction["Text Extraction & Cleaning"] --> summarization(["OpenAI Batch API\n(Summarization)"]) --> ranking["Ranking & Filtering"] --> embedding(["OpenAI Embedding API"]) --> Index[(
