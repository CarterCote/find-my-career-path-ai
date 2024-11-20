from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
from .database import get_db
from .config import get_settings
from .utils.job_search import JobSearchService
from .utils.job_retriever import JobSearchRetriever, SearchStrategy
from llama_index.core import Settings as LlamaSettings
from sentence_transformers import SentenceTransformer

# Global instance to maintain state
_job_search_service = None

def get_search_service(db: Session = Depends(get_db)) -> JobSearchService:
    """Get or create JobSearchService instance"""
    global _job_search_service
    
    if _job_search_service is None:
        print("\n========== DEPENDENCIES DEBUG ==========")
        print("1. Creating new JobSearchService instance")
        settings = get_settings()
        embed_model = SentenceTransformer(settings.embed_model_name)
        retriever = JobSearchRetriever(
            db=db, 
            embed_model=embed_model, 
            strategy=SearchStrategy.SEMANTIC
        )
        _job_search_service = JobSearchService(retriever=retriever, settings=settings, db=db)
        print("2. JobSearchService instance created")
    else:
        print("\n========== USING EXISTING JOBSEARCHSERVICE ==========")
        # Update DB connection for this request
        _job_search_service.db = db
        _job_search_service.retriever.db = db
    
    return _job_search_service