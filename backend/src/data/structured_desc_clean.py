import pandas as pd
import json
import re
import os

DEFAULT_JSON = {
    "Required skills": [],
    "Years of experience": "Not specified",
    "Education requirements": "Not specified",
    "Key responsibilities": [],
    "Benefits mentioned": [],
    "Required certifications": [],
    "Work environment": "Not specified"
}

def validate_and_format_json(json_string) -> str:
    # Handle non-string and empty values by falling back to DEFAULT_JSON
    if not isinstance(json_string, str) or json_string.strip() == "":
        return json.dumps(DEFAULT_JSON)
    
    # Remove any extraneous outer quotes
    json_string = json_string.strip('"')
    
    try:
        # Step 1: Quote unquoted keys
        json_string = re.sub(r'([{,])\s*([a-zA-Z0-9_ ]+)\s*:', r'\1"\2":', json_string)

        # Step 2: Escape unescaped double quotes within values
        json_string = re.sub(r'":\s*"([^"]*?)"', lambda m: '": "' + m.group(1).replace('"', '\\"') + '"', json_string)

        # Step 3: Parse JSON string to ensure validity
        parsed_json = json.loads(json_string)
        
        # Serialize back to a string with consistent JSON format
        return json.dumps(parsed_json, ensure_ascii=False, indent=4)
    
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error processing JSON: {e}")
        print(f"Problematic string: {json_string}")
        # Return DEFAULT_JSON if parsing fails
        return json.dumps(DEFAULT_JSON)


def fix_structured_description_in_csv(input_csv: str, output_csv: str):
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_csv, dtype=str, low_memory=False)
    
    if 'structured_description' in df.columns:
        # Apply the validation function to each entry
        df['structured_description'] = df['structured_description'].apply(validate_and_format_json)
    else:
        print("The 'structured_description' column does not exist in the CSV file.")
        return
    
    # Write the corrected DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Corrected CSV saved to {output_csv}")

if __name__ == "__main__":
    # Define input and output paths
    input_file = 'backend/data/checkpoints/enriched_data_checkpoint_12100.csv'
    output_file = 'backend/data/checkpoints/enriched_data_checkpoint_testing.csv'
    
    # Get absolute path of the input file
    absolute_path = os.path.abspath(input_file)
    print(f"Absolute path: {absolute_path}")

    # Execute the JSON fixing function
    fix_structured_description_in_csv(absolute_path, output_file)
