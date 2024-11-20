from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ...database import get_db
from ...dependencies import get_search_service

# Move these models to a new file later if they grow more complex
class JobSearchFilters(BaseModel):
    min_salary: Optional[float] = None
    skills: Optional[List[str]] = None
    work_environment: Optional[str] = None
    experience_years: Optional[str] = None
    education_level: Optional[str] = None

class JobSearchResult(BaseModel):
    id: int
    title: str
    company: str
    description: str
    salary_range: Optional[str] = None
    required_skills: List[str]
    experience_years: Optional[str] = None
    education_level: Optional[str] = None
    work_environment: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[JobSearchResult]
    session_id: Optional[str] = None

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"]
)

@router.get("/search", response_model=SearchResponse)
async def search_jobs(
    query: str,
    session_id: str = Query(default="default"),
    db: Session = Depends(get_db)
) -> Dict:
    """Direct job search endpoint for immediate results"""
    search_service = get_search_service(db)
    results = search_service.search(query, db, session_id)
    return {
        "results": results,
        "session_id": session_id
    }

@router.post("/search/filtered", response_model=SearchResponse)
async def filtered_search(
    filters: JobSearchFilters,
    db: Session = Depends(get_db)
) -> Dict:
    """Search with specific filters from structured descriptions"""
    search_service = get_search_service(db)
    results = search_service.filtered_search(
        min_salary=filters.min_salary,
        required_skills=filters.skills,
        work_environment=filters.work_environment,
        experience_years=filters.experience_years,
        education_level=filters.education_level
    )
    return {"results": results}