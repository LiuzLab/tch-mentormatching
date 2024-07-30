import os
import re
import pandas as pd

# pull the names from the table because it's more reliable to match this way
def load_mentor_data(csv_file='data/mentor_data.csv'):
    return pd.read_csv(csv_file)

def extract_mentor_id(mentor_summary):
    match = re.match(r'^(\d+)\.txt', mentor_summary)
    return match.group(1) if match else None

def extract_and_format_name(mentor_data):
    match = re.match(r'^##\s*(.+?)(?:\n\n|\Z)', mentor_data, re.DOTALL)
    if match:
        name = match.group(1).strip()
        return ' '.join(word.capitalize() for word in name.split())
    return "Unknown Name"

def clean_summary(summary):
    # Remove the file identifier and '=====' at the beginning
    cleaned = re.sub(r'^\d+\.txt\s*=+\s*', '', summary)
    return cleaned.strip()

def create_mentor_table_html(evaluated_matches):
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
    for match in evaluated_matches:
        mentor_summary = match["Mentor Summary"]
        mentor_id = extract_mentor_id(mentor_summary)
        
        # Find the matching row in mentor_data_df
        matching_row = mentor_data_df[mentor_data_df['Mentor_Profile'] == f"{mentor_id}.txt"]
        
        if not matching_row.empty:
            full_mentor_data = matching_row['Mentor_Data'].iloc[0]
            name = extract_and_format_name(full_mentor_data)
        else:
            name = "Unknown Name"

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

    return html