import os
import re


def clean_summary(summary):
    # Remove the "<numbers>.pdf =====" pattern and any leading/trailing whitespace
    cleaned = re.sub(r"^\d+\.pdf\s*=+\s*", "", summary)
    return cleaned.strip()


def extract_name(summary):
    # Clean the summary first
    cleaned_summary = clean_summary(summary)
    # Extract name from the cleaned summary (assumes name is at the beginning before "is")
    match = re.search(r"^(.+?)\s+is\s+", cleaned_summary)
    if match:
        return match.group(1)
    return "Unknown Name"  # Fallback if name can't be extracted


def create_mentor_table_html(evaluated_matches):
    # Get the directory of the current script (bin directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Read the HTML template
    template_path = os.path.join(
        project_root, "templates", "mentor_table_template.html"
    )
    with open(template_path, "r") as file:
        html_template = file.read()

    # Read the CSS file
    css_path = os.path.join(project_root, "static", "css", "mentor_table_styles.css")
    with open(css_path, "r") as file:
        css_content = file.read()

    # Create the mentor rows
    mentor_rows = ""
    for match in evaluated_matches:
        summary = clean_summary(match["Mentor Summary"])
        name = extract_name(match["Mentor Summary"])
        scores = match["Criterion Scores"]
        mentor_rows += f"""
        <tr>
            <td class="mentor-name" data-score="Similarity Score: {match['Similarity Score']}"><a href="https://profiles.viictr.org/display/{match['mentor_id']}">{name}</a></td>
            <td class="mentor-summary">
                <div class="summary-content">{summary}</div>
            </td>
            <td class="criterion-score">{scores['Research Interest']}</td>
            <td class="criterion-score">{scores['Availability']}</td>
            <td class="criterion-score">{scores['Skillset']}</td>
            <td class="criterion-score overall-score">{scores['Overall Match Quality']}</td>
        </tr>
        """

    # Create the full HTML with a scrollable container
    full_html = f"""
    <div class="table-container">
        <table class="mentor-table">
            <thead>
                <tr>
                    <th>Mentor Name</th>
                    <th>Profile Summary</th>
                    <th>Research Interest</th>
                    <th>Availability</th>
                    <th>Skillset</th>
                    <th>Overall Match</th>
                </tr>
            </thead>
            <tbody>
                {mentor_rows}
            </tbody>
        </table>
    </div>
    """

    # Insert the CSS and full HTML into the template
    html = html_template.format(table_content=full_html)
    html = html.replace("</head>", f"<style>{css_content}</style></head>")

    return html
