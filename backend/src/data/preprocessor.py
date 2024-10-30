from typing import List, Dict
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import re

class JobPostingPreprocessor:
    def __init__(self):
        # Download all required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
            
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire dataset
        """
        # Create copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text columns
        text_columns = ['title', 'description', 'skills_desc']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[f'processed_{col}'] = processed_df[col].apply(self.preprocess_text)
        
        # Categorize job titles
        processed_df['job_category'] = processed_df['title'].apply(self.categorize_job)
        
        # Process salary and numeric information
        numeric_cols = ['min_salary', 'med_salary', 'max_salary', 'closed_time', 'formatted_experience_level']
        for col in numeric_cols:
            if col in processed_df.columns:
                # Convert to numeric, replacing invalid values with None/NaN
                processed_df[col] = pd.to_numeric(
                    processed_df[col].apply(
                        lambda x: str(x).replace(',', '').strip() if pd.notna(x) else None
                    ), 
                    errors='coerce'
                )
                # Replace NaN with None (which will become NULL in SQL)
                processed_df[col] = processed_df[col].where(pd.notna(processed_df[col]), None)
        
        # Process location data
        if 'location' in processed_df.columns:
            processed_df['processed_location'] = processed_df['location'].apply(
                lambda x: x.strip('"\'') if pd.notna(x) else 'Unknown'
            )
            # Remove any internal quotes that might cause SQL issues
            processed_df['processed_location'] = processed_df['processed_location'].str.replace('"', '')
        
        # Process work type
        if 'formatted_work_type' in processed_df.columns:
            processed_df['processed_work_type'] = processed_df['formatted_work_type'].fillna('Unknown')
        
        # Convert remote_allowed from float to int, filling NA values with 0
        if 'remote_allowed' in processed_df.columns:
            processed_df['remote_allowed'] = processed_df['remote_allowed'].fillna(0).astype(int)
            
        return processed_df
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def categorize_job(self, title: str) -> str:
        """
        Categorize job titles
        """
        if pd.isna(title):
            return 'unknown'
            
        title_lower = title.lower()
        
        categories = {
            'engineering': ['engineer', 'developer', 'programmer', 'architect', 'devops'],
            'data_science': ['data scientist', 'machine learning', 'ai', 'analytics', 'data engineer'],
            'management': ['manager', 'director', 'lead', 'head', 'chief'],
            'design': ['designer', 'ux', 'ui', 'graphic'],
            'marketing': ['marketing', 'seo', 'content', 'social media'],
            'sales': ['sales', 'account executive', 'business development'],
            'finance': ['financial', 'accountant', 'analyst'],
            'hr': ['hr', 'human resources', 'recruiter', 'talent']
        }
        
        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
                
        return 'other'