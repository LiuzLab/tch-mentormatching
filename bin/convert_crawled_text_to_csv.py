import os
import csv
import glob

# Define the input directory and output file
input_directory = './data/txt_all/*.txt'
output_file = './data/mentor_data.csv'

# Get all txt files in the directory
txt_files = glob.glob(input_directory)

# Open the CSV file for writing
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow(['Mentor_Profile', 'Mentor_Data'])
    
    # Process each txt file
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                # Write the file name and content to the CSV file
                writer.writerow([os.path.basename(txt_file), content])
        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")

print(f"CSV file '{output_file}' has been created with the contents of all txt files.")
