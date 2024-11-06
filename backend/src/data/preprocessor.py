from typing import List, Dict
import pandas as pd

class JobPostingPreprocessor:
    def __init__(self):
        pass
        
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset by selecting and cleaning specific columns
        """
        # Define base columns (from raw data)
        base_columns = [
            'job_id',
            'company_name',
            'title',
            'description',
            'max_salary',
            'pay_period',
            'location',
            'company_id',
            'views',
            'med_salary',
            'min_salary',
            'description_embedding',
            'structured_description'
        ]
        
        # Create copy with base columns that exist in raw data
        raw_columns = base_columns[:-2]  # Exclude AI columns
        processed_df = df[raw_columns].copy()
        
        # Add AI enrichment columns in correct order
        processed_df['description_embedding'] = None
        processed_df['structured_description'] = None
        
        # Clean location data
        processed_df['location'] = processed_df['location'].fillna('Unknown')
        processed_df['location'] = processed_df['location'].str.strip().str.replace('"', '')
        
        # Process salary columns - ensure they're clean numbers
        salary_cols = ['min_salary', 'med_salary', 'max_salary']
        for col in salary_cols:
            processed_df[col] = pd.to_numeric(
                processed_df[col].astype(str).str.replace(',', ''),
                errors='coerce'
            ).fillna(0)
        
        # Clean company_id - ensure it's numeric
        processed_df['company_id'] = pd.to_numeric(processed_df['company_id'], errors='coerce').fillna(0)
        
        # Convert views to integer
        processed_df['views'] = pd.to_numeric(processed_df['views'], errors='coerce').fillna(0).astype(int)
        
        # Clean text fields
        text_cols = ['job_id', 'company_name', 'title', 'description', 'pay_period']
        for col in text_cols:
            processed_df[col] = processed_df[col].fillna('')
            processed_df[col] = processed_df[col].str.strip()
        
        # Update columns_to_keep for verification
        columns_to_keep = base_columns + ['description_embedding', 'structured_description']
        
        # Final verification that we only have the columns we want
        assert set(processed_df.columns) == set(columns_to_keep), "Extra columns found in processed dataframe"
        
        return processed_df