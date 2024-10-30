import pandas as pd
from pathlib import Path
from preprocessor import JobPostingPreprocessor
import csv

def main():
    try:
        # Setup paths
        data_dir = Path('backend/data')
        raw_dir = data_dir / 'raw'
        processed_dir = data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        print("Loading raw data...")
        df = pd.read_csv(raw_dir / 'postings.csv')
        print(f"Loaded {len(df)} records")
        
        # Initialize preprocessor
        print("Initializing preprocessor...")
        preprocessor = JobPostingPreprocessor()
        
        # Process data
        print("Processing data...")
        processed_df = preprocessor.preprocess_dataset(df)
        
        # Save processed data
        output_path = processed_dir / 'processed_jobs.csv'
        processed_df.to_csv(
            output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            na_rep='',
            float_format='%.2f'
        )
        print(f"Processed data saved to {output_path}")
        
        # Print processing summary
        print("\nProcessing Summary:")
        print(f"Original records: {len(df)}")
        print(f"Processed records: {len(processed_df)}")
        print(f"Unique job categories: {processed_df['job_category'].nunique()}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()