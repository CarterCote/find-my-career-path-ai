from typing import Dict, List
import openai
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer

class AIDescriptionEnricher:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        print("Initialized OpenAI with API key")
        # Initialize the local embedding model
        print("Loading local embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Local embedding model loaded successfully")
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using local Sentence Transformers model"""
        print("Generating embedding locally...")
        try:
            embedding = self.embedding_model.encode(text)
            embedding_list = embedding.tolist()  # Convert numpy array to list
            print(f"Embedding generated successfully (length: {len(embedding_list)})")
            return embedding_list
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_structured_description(self, description: str) -> Dict:
        """Create structured description using OpenAI's API"""
        print("Creating structured description via OpenAI...")
        prompt = f"""Please analyze this job description and extract key information in JSON format:
        
        Job Description:
        {description}
        
        Please provide:
        1. Required skills
        2. Years of experience
        3. Education requirements
        4. Key responsibilities
        5. Benefits mentioned
        6. Required certifications
        7. Work environment (remote/hybrid/onsite)
        """
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from job descriptions. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        print("Structured description created successfully")
        return response.choices[0].message.content
    
    def create_structured_descriptions_batch(self, descriptions: List[str]) -> List[Dict]:
        """Create structured descriptions for a batch of job descriptions"""
        print(f"Creating structured descriptions for {len(descriptions)} jobs...")
        
        # Construct the messages for the batch
        messages = []
        for desc in descriptions:
            messages.append({
                "role": "user",
                "content": f"""Please analyze this job description and extract key information in JSON format:
                Job Description: {desc}
                Please provide:
                1. Required skills
                2. Years of experience
                3. Education requirements
                4. Key responsibilities
                5. Benefits mentioned
                6. Required certifications
                7. Work environment (remote/hybrid/onsite)"""
            })
        
        # Make one API call for the batch
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from job descriptions. Respond only with valid JSON."}
            ] + messages,
            response_format={ "type": "json_object" }
        )
        
        return [response.choices[i].message.content for i in range(len(descriptions))]

def enrich_job_postings(df: pd.DataFrame, api_key: str, batch_size: int = 20) -> pd.DataFrame:
    """Enrich job postings with AI-generated content using batch processing"""
    print(f"\nStarting AI enrichment for {len(df)} records...")
    enricher = AIDescriptionEnricher(api_key)
    
    # Create new DataFrame with original columns plus new ones
    enriched_df = df.copy()
    enriched_df['description_embedding'] = None
    enriched_df['structured_description'] = None
    
    # Process in batches
    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]
        print(f"\n{'='*50}")
        print(f"Processing batch {start_idx//batch_size + 1}")
        print(f"Records {start_idx+1} to {min(start_idx+batch_size, len(df))}")
        
        try:
            # Generate all embeddings for batch (fast, local)
            print("\nGenerating embeddings for batch...")
            embeddings = enricher.embedding_model.encode(batch['description'].tolist())
            print(f"✓ Generated {len(embeddings)} embeddings")
            
            # Create all structured descriptions for batch (one API call)
            print("\nCreating structured descriptions for batch...")
            descriptions = batch['description'].tolist()
            structured_data = enricher.create_structured_descriptions_batch(descriptions)
            print(f"✓ Created {len(structured_data)} structured descriptions")
            
            # Update DataFrame with detailed logging
            for idx, (embedding, structured) in enumerate(zip(embeddings, structured_data)):
                df_idx = start_idx + idx
                row = batch.iloc[idx]
                print(f"\nProcessing record {df_idx + 1}/{len(df)}")
                print(f"Job Title: {row['title']}")
                
                enriched_df.at[df_idx, 'description_embedding'] = embedding.tolist()
                print(f"✓ Embedding saved (length: {len(embedding)})")
                
                enriched_df.at[df_idx, 'structured_description'] = structured
                print(f"✓ Structured description saved: {structured[:100]}...")
            
            # Save checkpoint after each batch
            checkpoint_path = f'backend/data/checkpoints/enriched_data_checkpoint_{start_idx+batch_size}.csv'
            enriched_df.to_csv(checkpoint_path, index=False)
            print(f"\n✓ Saved checkpoint at record {start_idx+batch_size}")
            
        except Exception as e:
            print(f"\n❌ Error processing batch: {str(e)}")
            error_checkpoint = f'backend/data/checkpoints/enriched_data_error_at_{start_idx}.csv'
            enriched_df.to_csv(error_checkpoint, index=False)
            print(f"Progress saved to {error_checkpoint}")
            continue
    
    # Verify the new columns exist and have data
    print("\nVerifying enriched data...")
    print(f"Columns in enriched DataFrame: {enriched_df.columns.tolist()}")
    print(f"Number of embeddings generated: {enriched_df['description_embedding'].notna().sum()}")
    print(f"Number of structured descriptions: {enriched_df['structured_description'].notna().sum()}")
    
    return enriched_df