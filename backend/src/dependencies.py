from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
from .database import get_db
from .config import get_settings
from .utils.job_search import JobSearchService
from .utils.job_retriever import JobSearchRetriever, SearchStrategy
from llama_index.core import Settings as LlamaSettings
from sentence_transformers import SentenceTransformer

def get_search_service(db: Session = Depends(get_db)) -> JobSearchService:
    settings = get_settings()
    embed_model = SentenceTransformer(settings.embed_model_name)
    retriever = JobSearchRetriever(
        db=db, 
        embed_model=embed_model, 
        strategy=SearchStrategy.SEMANTIC
    )
    return JobSearchService(retriever=retriever, settings=settings)