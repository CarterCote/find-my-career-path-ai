from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .utils.job_search import JobSearchService
from .utils.job_retriever import JobSearchRetriever
from .config import Settings
from .database import get_db

app = FastAPI()
settings = Settings()

@app.post("/search/jobs")
async def search_jobs(
    query: str,
    session_id: str = "default",
    db: Session = Depends(get_db)
):
    retriever = JobSearchRetriever(db=db, embed_model=settings.embedding_model)
    search_service = JobSearchService(retriever, settings)
    return search_service.search(query, session_id)