from __future__ import absolute_import, unicode_literals
from .celery import app
from typing import Dict, List
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..utils.job_search import JobSearchService, get_search_service

@app.task
def process_search_results(results: List[Dict], query: str, session_id: str) -> List[Dict]:
    """Background task to process and rank search results"""
    db = SessionLocal()
    try:
        search_service = get_search_service(db)
        filtered_results = search_service.filter_irrelevant_jobs(results, query)
        return filtered_results
    finally:
        db.close()

@app.task
def profile_based_search(profile_data: Dict, query: str, session_id: str) -> Dict:
    """Background task for profile-based job search"""
    db = SessionLocal()
    try:
        search_service = get_search_service(db)
        results = search_service.profile_based_search(profile_data, query, db)
        return results
    finally:
        db.close()

@app.task
def generate_job_recommendations(session_id: str) -> List[Dict]:
    """Background task to generate personalized job recommendations"""
    db = SessionLocal()
    try:
        search_service = get_search_service(db)
        user_profile = db.query(UserProfile).filter(
            UserProfile.session_id == session_id
        ).first()
        
        if not user_profile:
            return []
            
        results = search_service.verify_and_rank_results(
            search_service.search(
                query="",
                db=db,
                session_id=session_id
            ).get("jobs", []),
            user_profile
        )
        return results
    finally:
        db.close()