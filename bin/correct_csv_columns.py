import pandas as pd
import os

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def create_column_name_mapping():
    """
    Create a dictionary for column name mapping.
    
    Returns:
    dict: Mapping of old column names to new column names.
    """
    return {
        "Mentor Profile": "Mentor_Profile",
        "Mentee Profile": "Mentee_Profile",
        "Mock Student CV": "Mock_Student_CV",
        "PDF Text": "PDF_Text",
        "Mentor Summary": "Mentor_Summary",
        "Mentee Summary": "Mentee_Summary"
    }

def correct_column_names(df, column_mapping):
    """
    Correct column names in a DataFrame based on a mapping.
    
    Args:
    df (pd.DataFrame): Input DataFrame.
    column_mapping (dict): Mapping of old column names to new column names.
    
    Returns:
    pd.DataFrame: DataFrame with corrected column names.
    """
    return df.rename(columns=column_mapping)

def save_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.
    
    Args:
    df (pd.DataFrame): DataFrame to save.
    file_path (str): Path where the CSV file will be saved.
    """
    df.to_csv(file_path, index=False)
    print(f"Corrected DataFrame saved to: {file_path}")

def correct_csv_columns(input_file, output_file):
    """
    Correct column names in a CSV file and save the result.
    
    Args:
    input_file (str): Path to the input CSV file.
    output_file (str): Path where the output CSV file will be saved.
    """
    try:
        df = load_csv(input_file)
        column_mapping = create_column_name_mapping()
        df_corrected = correct_column_names(df, column_mapping)
        save_csv(df_corrected, output_file)
        print("Column names corrected successfully.")
    except Exception as e:
        print(f"An error occurred while correcting CSV columns: {str(e)}")

def main():
    input_file = "../simulated_data/mentor_student_cvs_with_summaries_final.csv"
    output_file = "../simulated_data/mentor_student_cvs_with_summaries_final_corrected.csv"
    correct_csv_columns(input_file, output_file)

if __name__ == "__main__":
    main()
