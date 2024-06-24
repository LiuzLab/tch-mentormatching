# TCH Mentor-Matching

## Building Database 

```mermaid
  flowchart LR;
     mentor_cv[/"Mentor CV (PDF)"/] --> ChatGPT(["ChatGPT\n(Summarization)"]) --> ChatGPT_Embedding(["ChatGPT\n(Embedding)"]) --> Index[("FAISS indexer")]
```

## Mentor Search for Mentee

```mermaid
  flowchart LR;
     mentee[/"Mentee CV (PDF)"/] --> ChatGPT(["ChatGPT\n(Summarization)"]) --> ChatGPT_Embedding(["ChatGPT\n(Embedding)"]) --> Index[("Query to\nFAISS indexer")] --> Top_Candidate(["Search Top-K\nrelevant Mentors"]) --> ChatGPT_Evaluation(["Evaulation\nFor each relevant mentor"]) --> ChatGPT_Matching[/"Output the matching result"/]
```
