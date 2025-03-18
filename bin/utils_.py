import re
import pandas as pd
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

# use this to add a Professor_Type metadata column in the .csv file; allows us to search for
# only professors of a specific typke
def find_professor_type(mentor_data):
    """
    input: pandas dataframe (y['Mentor_Data'])
    output: adjusted column (y['Professor_Type'])
    usage: y['Professor_Type'] = y['Mentor_Data'].apply(find_professor_type)
    """
    # Extract the title using regex
    title_match = re.search(r'Title\|(.*?)(?:\n|$)', mentor_data)
    if title_match:
        title = title_match.group(1).strip().lower()
        if 'distinguished' in title:
            return 'Distinguished Professor'
        elif 'associate professor' in title:
            return 'Associate Professor'
        elif 'assistant professor' in title:
            return 'Assistant Professor'
        elif 'adjunct professor' in title:
            return 'Adjunct Professor'
        elif 'professor' in title:
            return 'Professor'
        elif 'instructor' in title:
            return 'Instructor'
        elif 'clinical' in title:
            return 'Clinical Professor'
        else:
            return title.capitalize()  # Return the title as-is if it doesn't match known categories
    return 'Unknown'

def rank_professors(df, professor_type_column='Professor_Type', rank_column='Rank'):
    # Define the ranking dictionary
    rank_mapping = {
        'Chair': 5,
        'Distinguished Professor': 4,
        'Professor': 3,
        'Associate Professor': 2,
        'Assistant Professor': 1,
        'Adjunct Professor': -1
    }
    
    # Function to assign rank based on Professor Type
    def assign_rank(professor_type):
        return rank_mapping.get(professor_type, -2)  # Default to -2 for Unknown types
    
    # Apply the ranking
    df[rank_column] = df[professor_type_column].apply(assign_rank)
    
    return df
