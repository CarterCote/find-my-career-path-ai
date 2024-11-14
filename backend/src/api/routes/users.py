from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from ...database import get_db
from ...crud import create_user_profile, get_user_profile
from ...utils.job_search import get_search_service
from ...dependencies import get_settings
from ...utils.job_retriever import SearchStrategy

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

class UserPreferences(BaseModel):
    session_id: str
    core_values: List[str]
    work_culture: List[str]
    skills: List[str]
    additional_interests: Optional[str] = None

class JobRecommendation(BaseModel):
    job_id: str
    title: str
    company_name: str
    description: str
    salary_range: Optional[str] = None
    match_score: float
    matching_skills: List[str]
    matching_culture: List[str]
    location: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[JobRecommendation]
    session_id: str

@router.post("/preferences", response_model=RecommendationResponse)
async def create_user_preferences(
    preferences: UserPreferences,
    db: Session = Depends(get_db)
):
    try:
        # Create or update user profile
        user_profile = create_user_profile(
            db=db,
            session_id=preferences.session_id
        )
        
        # Update user preferences
        user_profile.core_values = preferences.core_values
        user_profile.work_culture = preferences.work_culture
        user_profile.skills = preferences.skills
        user_profile.additional_interests = preferences.additional_interests
        
        db.commit()
        db.refresh(user_profile)
        
        # Get search service with hybrid strategy
        search_service = get_search_service(db)
        search_service.retriever.strategy = SearchStrategy.HYBRID
        
        # Create initial filters
        filters = {
            "required_skills": preferences.skills,
            "work_environment": preferences.work_culture[0] if preferences.work_culture else None,
            "limit": 20  # Adjust as needed
        }
        
        # Construct context query from preferences
        context_query = f"""
        Looking for jobs that match these skills: {', '.join(preferences.skills)}
        with work culture preferences: {', '.join(preferences.work_culture)}
        and values: {', '.join(preferences.core_values)}
        Additional context: {preferences.additional_interests or ''}
        """
        
        # Use hybrid search
        results = search_service.search(
            query=context_query,
            db=db,
            session_id=preferences.session_id
        )
        
        # Combine and rank results
        all_results = []
        seen_jobs = set()
        
        # Process filtered results first
        for job in results.get("jobs", []):
            if job["job_id"] not in seen_jobs:
                seen_jobs.add(job["job_id"])
                
                # Calculate match scores
                matching_skills = [
                    skill for skill in preferences.skills 
                    if skill.lower() in job.get("required_skills", [])
                ]
                
                matching_culture = [
                    culture for culture in preferences.work_culture 
                    if culture.lower() in job.get("work_environment", "").lower()
                ]
                
                match_score = (
                    len(matching_skills) / len(preferences.skills) * 0.6 +
                    len(matching_culture) / len(preferences.work_culture) * 0.4
                )
                
                all_results.append(JobRecommendation(
                    job_id=job["job_id"],
                    title=job["title"],
                    company_name=job["company_name"],
                    description=job["description"],
                    salary_range=f"${job.get('min_salary', 0):,.0f} - ${job.get('max_salary', 0):,.0f}",
                    match_score=match_score,
                    matching_skills=matching_skills,
                    matching_culture=matching_culture,
                    location=job.get("location")
                ))
        
        # Sort by match score and take top 10
        all_results.sort(key=lambda x: x.match_score, reverse=True)
        top_recommendations = all_results[:10]
        
        return RecommendationResponse(
            recommendations=top_recommendations,
            session_id=preferences.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
def read_user(session_id: str, db: Session = Depends(get_db)):
    return get_user_profile(db=db, session_id=session_id)