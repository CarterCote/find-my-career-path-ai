import pandas as pd
from pathlib import Path
from preprocessor import JobPostingPreprocessor

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
        
        # Convert extracted_skills lists to PostgreSQL array format
        processed_df['extracted_skills'] = processed_df['extracted_skills'].apply(
            lambda x: '{' + ','.join(x) + '}' if isinstance(x, list) and len(x) > 0 else '{}'
        )
        
        # Save processed data
        output_path = processed_dir / 'processed_jobs.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        # Print processing summary
        print("\nProcessing Summary:")
        print(f"Original records: {len(df)}")
        print(f"Processed records: {len(processed_df)}")
        print(f"Unique job categories: {processed_df['job_category'].nunique()}")
        print(f"Total skills extracted: {processed_df['extracted_skills'].str.len().sum()}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()