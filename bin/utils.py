import re
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
