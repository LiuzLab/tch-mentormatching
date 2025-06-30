import os
import re
import pandas as pd
from ..utils import clean_summary, extract_and_format_name 
from ..config.paths import PATH_TO_MENTOR_DATA

def load_mentor_data(csv_file=PATH_TO_MENTOR_DATA):
    return pd.read_csv(csv_file)

# def extract_mentor_id(mentor_summary):
#     match = re.match(r'^(\d+)\.txt', mentor_summary)
#     return match.group(1) if match else None

def extract_mentor_id(mentor_summary, metadata=None):
    # If metadata is provided and has Mentor_Profile, use that
    if metadata and 'Mentor_Profile' in metadata:
        # Extract just the number part from something like '1826469.txt'
        profile_name = metadata['Mentor_Profile']
        return profile_name.replace('.txt', '')
    
    # Fallback to the old method
    match = re.match(r'^(\d+)\.txt', mentor_summary)
    return match.group(1) if match else None

def create_mentor_table_html_and_csv_data(evaluated_matches):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    template_path = os.path.join(project_root, "templates", "mentor_table_template.html")
    with open(template_path, "r") as file:
        html_template = file.read()

    css_path = os.path.join(project_root, "static", "css", "mentor_table_styles.css")
    with open(css_path, "r") as file:
        css_content = file.read()
        
    mentor_data_df = load_mentor_data()

    mentor_rows = ""
    csv_data = []

    for match in evaluated_matches:
        print("match", match)
        mentor_summary = match["Mentor Summary"]
        #mentor_id = extract_mentor_id(mentor_summary)
        mentor_id = extract_mentor_id(mentor_summary, match.get('metadata', {}))
        
        # Find the matching row in mentor_data_df
        matching_row = mentor_data_df[mentor_data_df['Mentor_Profile'] == f"{mentor_id}.txt"]
        
        if not matching_row.empty:
            full_mentor_data = matching_row['Mentor_Data'].iloc[0]
            name = extract_and_format_name(full_mentor_data)
        else:
            name = "Unknown Name"
            full_mentor_data = "No profile data available"

        cleaned_summary = clean_summary(mentor_summary)
        scores = match["Criterion Scores"]
        evaluation_summary = scores.get('Evaluation Summary', 'No evaluation summary available')

        mentor_rows += f"""
        <tr>
            <td class="mentor-name" data-score="Similarity Score: {match['Similarity Score']}"><a href="https://profiles.viictr.org/display/{mentor_id}">{name}</a></td>
            <td class="mentor-summary">
                <div class="summary-content">{cleaned_summary}</div>
            </td>
            <td class="evaluation-summary">
                <div class="summary-content">{evaluation_summary}</div>
            </td>
        </tr>
        """

        mentor_id = match["Mentor Summary"].split("===")[0].strip().replace(".txt", "")
        mentor_name = extract_and_format_name(match["Mentor Summary"])
        mentor_summary = clean_summary(match["Mentor Summary"])
        evaluation_summary = match["Criterion Scores"]["Evaluation Summary"]
        overall_score = match["Criterion Scores"]["Overall Match Quality"]
        research_score = match["Criterion Scores"]["Research Interest"]
        availability_score = match["Criterion Scores"]["Availability"]
        skillset_score = match["Criterion Scores"]["Skillset"]
        #similarity_score = match["Similarity Score"]

        csv_data.append({
            "Mentor Name": name,
            "Mentor ID": mentor_id,
            "Mentor Summary": mentor_summary,
            "Evaluation Summary": evaluation_summary,
            "Overall Match Quality": overall_score,
            "Research Interest Score": research_score,
            "Availability Score": availability_score,
            "Skillset Score": skillset_score,
            #"Similarity Score": similarity_score #leave this out for now
        })


    full_html = f"""
    <div class="table-container">
        <table class="mentor-table">
            <thead>
                <tr>
                    <th>Mentor Name</th>
                    <th>Profile Summary</th>
                    <th>Evaluation Summary</th>
                </tr>
            </thead>
            <tbody>
                {mentor_rows}
            </tbody>
        </table>
    </div>
    """

    html = html_template.format(table_content=full_html)
    html = html.replace("</head>", f"<style>{css_content}</style></head>")

    return html, csv_data
