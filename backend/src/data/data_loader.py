from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import kagglehub
import os

class CareerDataLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data loader
        Args:
            cache_dir: Optional directory to store downloaded dataset
        """
        self.cache_dir = 'backend/data/raw'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.dataset_path = None
    
    def download_dataset(self) -> str:
        """
        Download the LinkedIn job postings dataset from Kaggle
        Returns:
            Path to the downloaded dataset
        """
        self.dataset_path = kagglehub.dataset_download(
            "arshkon/linkedin-job-postings"
        )
        return self.dataset_path
        
    def load_job_postings(self) -> pd.DataFrame:
        """
        Load and perform initial cleaning of job postings dataset
        """
        if not self.dataset_path:
            self.download_dataset()
            
        # The dataset likely contains multiple CSV files, find the main one
        csv_files = list(Path(self.dataset_path).glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset") 