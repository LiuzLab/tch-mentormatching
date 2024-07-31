import os
import pandas as pd
import re
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from .generate_text import generate_text_async


# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=api_key)

# Instructions for evaluating the mentor-mentee pair
instructions = (
    "Evaluate the mentor-mentee pair for the following criteria and provide a score for each as well as an overall match quality score:\n\n"
    "First extract the following information:\n"
    "**Mentor:**\n"
    "- Name: {mentor_name}\n"
    "- Position: {mentor_position}\n"
    "- Research Interests: {mentor_research_interests}\n\n"
    "**Mentee:**\n"
    "- Name: {mentee_name}\n"
    "- Skillset: {mentee_skillset}\n"
    "- Research Interests: {mentee_research_interests}\n\n"
    "Next, evaluate the pair given the following criteria and generate an overall match quality score between 1 and 10:\n"
    "1. Research Interest: Do both have the same research interest?\n"
    "2. Availability: Does the mentor have the capability for mentorship based on their position (higher for assistant professor, lower otherwise)?\n"
    "3. Skillset: Does the mentee have a proper skillset relevant to the mentor's research?\n\n"
    "Provide a score (1-10) for each criterion and an overall match quality score.\n\n"
)


async def evaluate_pair_with_llm(
    client, mentor_summary, mentee_summary, instructions=instructions
):
    """
    Evaluate the mentor-mentee pair using the LLM.

    Args:
        client (OpenAI): OpenAI client instance.
        mentor_summary (str): Summary information for the mentor.
        mentee_summary (str): Summary information for the mentee.
        instructions (str): Evaluation instructions for the LLM.

    Returns:
        str: Evaluation text generated by the LLM.
    """
    # Append mentor and mentee summaries to the instructions
    combined_instructions = (
        instructions
        + f"\n\nMentor Summary:\n{mentor_summary}\n\nMentee Summary:\n{mentee_summary}"
    )

    # Generate the evaluation text using the LLM
    evaluation_text = await generate_text_async(client, "", combined_instructions)

    return evaluation_text

async def extract_eval_scores_with_llm(client, evaluation_text):
    """
    Extract evaluation scores and summary from the evaluation text using GPT-4.

    Args:
        client (OpenAI): OpenAI client instance.
        evaluation_text (str): Evaluation text generated by the LLM.

    Returns:
        dict: Extracted scores and summary for research interest, availability, skillset, and overall match quality.
    """
    # Prepare the prompt for GPT-4
    prompt = (
        "## Task: Extract the four scores and provide a structured evaluation summary from this text. Follow this example formatting:\n\n"
        "Output Format:\n"
        "Research Interest: <score>\n"
        "Availability: <score>\n"
        "Skillset: <score>\n"
        "Overall Match: <score>\n\n"
        "Evaluation Summary:\n" 
        "## Text:\n"
        + evaluation_text
    )

    # Generate the scores and summary text using the LLM
    structured_evaluation = await generate_text_async(client, "", prompt)

    # Extract scores using regex
    research_score = re.search(r"Research Interest: (\d+)", structured_evaluation)
    availability_score = re.search(r"Availability: (\d+)", structured_evaluation)
    skillset_score = re.search(r"Skillset: (\d+)", structured_evaluation)
    overall_score = re.search(r"Overall Match: (\d+(\.\d+)?)", structured_evaluation)

    # Convert to integers or floats, or None if not found
    research_score = int(research_score.group(1)) if research_score else None
    availability_score = int(availability_score.group(1)) if availability_score else None
    skillset_score = int(skillset_score.group(1)) if skillset_score else None
    overall_score = float(overall_score.group(1)) if overall_score else None

    # Extract evaluation summary
    summary_match = re.search(r"Evaluation Summary:(.*?)$", structured_evaluation, re.DOTALL)
    evaluation_summary = summary_match.group(1).strip() if summary_match else "No summary available"

    return {
        "Overall Match Quality": overall_score,
        "Research Interest": research_score,
        "Availability": availability_score,
        "Skillset": skillset_score,
        "Evaluation Summary": evaluation_summary
    }

if __name__ == "__main__":
    # Test usage
    mentor_summary_example = "Mentor: Dr. John Doe, Associate Professor, Research Interests: Artificial Intelligence, Natural Language Processing, Dog Food Nutrition.."
    mentee_summary_example = "Mentee: Jane Smith, Skills: Python, Data Analysis, Research Interests: Ancient Literature, Navajo Linguistics, Byzantine Pottery."

    evaluation = evaluate_pair_with_llm(
        client, mentor_summary_example, mentee_summary_example
    )
    print(evaluation)

    overall_score, research_score, availability_score, skillset_score, evaluation_summary = extract_eval_scores_with_llm(
        evaluation
    )
    scores = extract_eval_scores_with_llm(client, evaluation)
    scores_df = pd.DataFrame([scores])
    print(scores_df.head())
