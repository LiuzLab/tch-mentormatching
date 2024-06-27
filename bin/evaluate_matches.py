import os
import pandas as pd
import re
from dotenv import load_dotenv
from openai import OpenAI
from .generate_text import generate_text

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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


def evaluate_pair_with_llm(
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
    evaluation_text = generate_text(client, "", combined_instructions)

    return evaluation_text

def extract_eval_scores_with_llm(client, evaluation_text):
    """
    Extract evaluation scores from the evaluation text using GPT-4.

    Args:
        client (OpenAI): OpenAI client instance.
        evaluation_text (str): Evaluation text generated by the LLM.

    Returns:
        dict: Extracted scores for research interest, availability, skillset, and overall match quality.
    """
    # Prepare the prompt for GPT-4
    prompt = (
        "## Task: Extract the four scores in this text and follow this example formatting:\n\n"
        "Output Format:\n"
        "Research Interest: <score>\n"
        "Availability: <score>\n"
        "Skillset: <score>\n"
        "Overall Match: <score>\n\n"
        "## Text:\n"
        + evaluation_text
    )

    # Generate the scores text using the LLM
    scores_text = generate_text(client, "", prompt)

    # Extract scores using regex
    research_score = re.search(r"Research Interest: (\d+)", scores_text)
    availability_score = re.search(r"Availability: (\d+)", scores_text)
    skillset_score = re.search(r"Skillset: (\d+)", scores_text)
    overall_score = re.search(r"Overall Match: (\d+(\.\d+)?)", scores_text)

    # Convert to integers or floats, or None if not found
    research_score = int(research_score.group(1)) if research_score else None
    availability_score = int(availability_score.group(1)) if availability_score else None
    skillset_score = int(skillset_score.group(1)) if skillset_score else None
    overall_score = float(overall_score.group(1)) if overall_score else None

    return {
        "Research Interest": research_score,
        "Availability": availability_score,
        "Skillset": skillset_score,
        "Overall Match Quality": overall_score,
    }

if __name__ == "__main__":
    # Test usage
    mentor_summary_example = "Mentor: Dr. John Doe, Associate Professor, Research Interests: Artificial Intelligence, Natural Language Processing, Dog Food Nutrition.."
    mentee_summary_example = "Mentee: Jane Smith, Skills: Python, Data Analysis, Research Interests: Ancient Literature, Navajo Linguistics, Byzantine Pottery."

    evaluation = evaluate_pair_with_llm(
        client, mentor_summary_example, mentee_summary_example
    )
    print(evaluation)

    research_score, availability_score, skillset_score, overall_score = extract_scores(
        evaluation
    )
    scores = extract_scores(evaluation)
    scores_df = pd.DataFrame([scores])
    print(scores_df.head())
