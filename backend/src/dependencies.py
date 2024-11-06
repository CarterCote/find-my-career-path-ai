from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
from .database import get_db
from .utils.job_search import JobSearchService
from .utils.job_retriever import JobSearchRetriever
from .config import Settings

def get_settings() -> Settings:
    return Settings()

def get_search_service(db: Session = Depends(get_db)) -> JobSearchService:
    settings = get_settings()
    retriever = JobSearchRetriever(db, settings.embed_model)
    return JobSearchService(retriever, settings)