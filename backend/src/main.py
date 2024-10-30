from fastapi import Depends, FastAPI
from .utils.job_search import JobSearchService
from .utils.job_retriever import JobSearchRetriever
from .config import Settings
from sqlalchemy.orm import Session
from . import crud, models
from .database import get_db

app = FastAPI()
settings = Settings()

# Initialize once at startup
retriever = JobSearchRetriever()
search_service = JobSearchService(retriever, settings)

@app.post("/search/jobs")
async def search_jobs(query: str, session_id: str = "default"):
    """Direct job search endpoint for immediate results"""
    return search_service.search(query, session_id)

@app.post("/chat")
async def chat(message: str = "", session_id: str = "default"):
    """Conversational endpoint for interactive job search"""
    return search_service.chat(message, session_id)

@app.post("/users/")
def create_user(session_id: str, db: Session = Depends(get_db)):
    return crud.create_user_profile(db=db, session_id=session_id)

@app.get("/users/{session_id}")
def read_user(session_id: str, db: Session = Depends(get_db)):
    return crud.get_user_profile(db=db, session_id=session_id)