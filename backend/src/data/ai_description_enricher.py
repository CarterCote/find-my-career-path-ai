from typing import Dict, List
import openai
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import json

class AIDescriptionEnricher:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        print("Initialized OpenAI with API key: " + openai.api_key)
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
        """Create structured description using OpenAI's API with length handling"""
        try:
            # Truncate description if too long (roughly 4000 tokens ~ 16000 chars)
            max_chars = 16000
            if len(description) > max_chars:
                description = description[:max_chars] + "..."
            
            prompt = f"""Please analyze this job description and extract key information in JSON format:
            
            Job Description:
            {description}
            
            Please provide:
            1. Required skills (list)
            2. Years of experience (text)
            3. Education requirements (text)
            4. Key responsibilities (list)
            5. Benefits mentioned (list)
            6. Required certifications (list)
            7. Work environment (remote/hybrid/onsite)
            
            Keep the response concise and focused on the most important points.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o",  # Using 3.5 to save costs
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from job descriptions. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                max_tokens=1000  # Limit response size
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in create_structured_description: {str(e)}")
            # Return a basic structure if there's an error
            return json.dumps({
                "Required skills": [],
                "Years of experience": "Not specified",
                "Education requirements": "Not specified",
                "Key responsibilities": [],
                "Benefits mentioned": [],
                "Required certifications": [],
                "Work environment": "Not specified"
            })
    
    def create_structured_descriptions_batch(self, descriptions: List[str], max_batch_size: int = 10) -> List[Dict]:
        """Create structured descriptions for a batch of job descriptions"""
        all_results = []
        
        # Process in smaller sub-batches for API calls
        for i in range(0, len(descriptions), max_batch_size):
            sub_batch = descriptions[i:i+max_batch_size]
            print(f"Processing sub-batch {i//max_batch_size + 1} of {len(descriptions)//max_batch_size + 1}")
            
            # Process each description individually (more reliable)
            for desc in sub_batch:
                try:
                    structured_desc = self.create_structured_description(desc)
                    all_results.append(structured_desc)
                except Exception as e:
                    print(f"Error processing description: {str(e)}")
                    # Add empty result for failed item
                    all_results.append(json.dumps({
                        "Required skills": [],
                        "Years of experience": "Not specified",
                        "Education requirements": "Not specified",
                        "Key responsibilities": [],
                        "Benefits mentioned": [],
                        "Required certifications": [],
                        "Work environment": "Not specified"
                    }))
        
        return all_results

def enrich_job_postings(df: pd.DataFrame, api_key: str, batch_size: int = 20) -> pd.DataFrame:
    print(f"\nStarting AI enrichment for {len(df)} records...")
    enricher = AIDescriptionEnricher(api_key)
    enriched_df = df.copy()
    
    # Initialize columns
    if 'description_embedding' not in enriched_df.columns:
        enriched_df['description_embedding'] = None
    if 'structured_description' not in enriched_df.columns:
        enriched_df['structured_description'] = None
    
    # Process in batches
    for i in range(0, len(enriched_df), batch_size):
        batch = enriched_df.iloc[i:i+batch_size].copy()
        
        try:
            # Generate embeddings in batch (this is already fast as it's local)
            print(f"Generating embeddings for batch {i//batch_size + 1}...")
            embeddings = enricher.embedding_model.encode(batch['description'].tolist())
            for j, idx in enumerate(batch.index):
                enriched_df.at[idx, 'description_embedding'] = embeddings[j].tolist()
            
            # Use batch API call for structured descriptions
            print(f"Creating structured descriptions for batch {i//batch_size + 1}...")
            structured_descriptions = enricher.create_structured_descriptions_batch(batch['description'].tolist())
            for j, idx in enumerate(batch.index):
                enriched_df.at[idx, 'structured_description'] = structured_descriptions[j]
            
            # Save checkpoint
            checkpoint_path = f'backend/data/checkpoints/enriched_data_checkpoint_{i+batch_size}.csv'
            enriched_df.to_csv(checkpoint_path, index=False)
            print(f"✓ Saved checkpoint at record {i+batch_size}")
            
        except Exception as e:
            print(f"❌ Error in batch {i//batch_size + 1}: {str(e)}")
            error_path = f'backend/data/checkpoints/enriched_data_error_at_{i}.csv'
            enriched_df.to_csv(error_path, index=False)
            continue
    
    return enriched_df