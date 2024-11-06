import pandas as pd
from pathlib import Path
from preprocessor import JobPostingPreprocessor
import csv
from ai_description_enricher import enrich_job_postings
import os
from dotenv import load_dotenv
from utils.job_graph import JobGraph
import pickle

def process_data(df: pd.DataFrame, openai_api_key: str = None) -> pd.DataFrame:
    """Process data with cleaning, AI enrichment, and graph building"""
    print("Starting data processing...")
    
    # Use the preprocessor for basic cleaning
    preprocessor = JobPostingPreprocessor()
    processed_df = preprocessor.preprocess_dataset(df)
    print(f"Basic cleaning complete. Columns: {processed_df.columns.tolist()}")
    print(openai_api_key)
    print(openai_api_key)
    # AI enrichment after cleaning if API key provided
    if openai_api_key:
        print("\nStarting AI enrichment phase...")
        # Process in batches to manage API usage
        processed_chunks = []
        for i in range(0, len(processed_df), 100):
            chunk = processed_df[i:i+100].copy()
            
            # Here we'll add:
            # 1. Generate embeddings for descriptions
            # 2. Create structured descriptions using OpenAI
            print(f"\nProcessing batch {i//100 + 1}...")

            enriched_chunk = enrich_job_postings(chunk, openai_api_key)
            processed_chunks.append(enriched_chunk)
            print(f"Completed batch {i//100 + 1}")
        
        processed_df = pd.concat(processed_chunks, ignore_index=True)
        print("\nAI enrichment complete!")
        print(f"Final columns: {processed_df.columns.tolist()}")
    
    # Build and save the job graph
    job_graph = JobGraph()
    graph = job_graph.build_graph(processed_df)
    
    # Save graph for later use
    with open('backend/data/job_graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    
    return processed_df

def main():
    # Load environment variables at the start
    load_dotenv(Path('backend/.env'))
    
    try:
        # Setup paths
        data_dir = Path('backend/data')
        raw_dir = data_dir / 'raw'
        processed_dir = data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        print("\nLoading raw data...")
        df = pd.read_csv(
            raw_dir / 'postings.csv',
            escapechar='\\',
            quoting=csv.QUOTE_MINIMAL,
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False
        )
        print(f"Loaded {len(df)} records")
        
        # Get API key and verify it's loaded
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables!")
        print("OpenAI API key loaded successfully")
        
        # Process data with API key
        processed_df = process_data(df, openai_api_key=openai_api_key)
        
        # Save processed data
        output_path = processed_dir / 'processed_jobs.csv'
        print("\nSaving processed data...")
        print(f"Columns being saved: {processed_df.columns.tolist()}")
        processed_df.to_csv(
            output_path,
            index=False,
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            escapechar=None,
            lineterminator='\n',
            encoding='utf-8'
        )
        
        # Verify the output
        print("\nVerifying output file...")
        test_df = pd.read_csv(output_path, quoting=csv.QUOTE_ALL)
        print(f"Verification successful: {len(test_df)} records")
        print("Columns in processed file:")
        print(test_df.columns.tolist())
        
        # Print sample of enriched data
        if 'structured_description' in test_df.columns:
            print("\nSample structured description:")
            print(test_df['structured_description'].iloc[0])
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()