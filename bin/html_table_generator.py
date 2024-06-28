import os
import re

def clean_summary(summary):
    cleaned = re.sub(r"^\d+\.pdf\s*=+\s*", "", summary)
    return cleaned.strip()

def extract_name(summary):
    cleaned_summary = clean_summary(summary)
    match = re.search(r"^(.+?)\s+is\s+", cleaned_summary)
    if match:
        return match.group(1)
    return "Unknown Name"

def create_mentor_table_html(evaluated_matches):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    template_path = os.path.join(project_root, "templates", "mentor_table_template.html")
    with open(template_path, "r") as file:
        html_template = file.read()

    css_path = os.path.join(project_root, "static", "css", "mentor_table_styles.css")
    with open(css_path, "r") as file:
        css_content = file.read()

    mentor_rows = ""
    for match in evaluated_matches:
        summary = clean_summary(match["Mentor Summary"])
        name = extract_name(match["Mentor Summary"])
        scores = match["Criterion Scores"]
        evaluation_summary = scores.get('Evaluation Summary', 'No evaluation summary available')
        mentor_rows += f"""
        <tr>
            <td class="mentor-name" data-score="Similarity Score: {match['Similarity Score']}"><a href="https://profiles.viictr.org/display/{match['mentor_id']}">{name}</a></td>
            <td class="mentor-summary">
                <div class="summary-content">{summary}</div>
            </td>
            <td class="evaluation-summary">
                <div class="summary-content">{evaluation_summary}</div>
            </td>
            <td class="criterion-score overall-score">{scores['Overall Match Quality']}</td>
            <td class="criterion-score">{scores['Research Interest']}</td>
            <td class="criterion-score">{scores['Availability']}</td>
            <td class="criterion-score">{scores['Skillset']}</td>
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
                    <th>Overall Match</th>
                    <th>Research Interest</th>
                    <th>Availability</th>
                    <th>Skillset</th>
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