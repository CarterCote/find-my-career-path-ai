from typing import Dict
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from src.database import get_db
from src.services.job_search import get_search_service
from src.models.job_search import JobSearchFilters

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"]
)

@router.get("/search")
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

@router.post("/search/filtered")
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