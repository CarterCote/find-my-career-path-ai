from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from ...database import get_db
from ...crud import create_user_profile, get_user_profile
from ...dependencies import get_search_service
from ...dependencies import get_settings
from ...utils.job_retriever import SearchStrategy

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# Request/Response Models
class UserPreferences(BaseModel):
    session_id: str
    core_values: List[str]
    work_culture: List[str]
    skills: List[str]
    additional_interests: Optional[str] = None

class ProfileResponse(BaseModel):
    message: str
    session_id: str

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

# Endpoints
@router.post("/preferences", response_model=ProfileResponse)
async def create_user_preferences(
    preferences: UserPreferences,
    db: Session = Depends(get_db)
):
    """Create or update user profile with hard constraints"""
    try:
        user_profile = create_user_profile(
            db=db,
            session_id=preferences.session_id
        )
        
        user_profile.core_values = preferences.core_values
        user_profile.work_culture = preferences.work_culture
        user_profile.skills = preferences.skills
        user_profile.additional_interests = preferences.additional_interests
        
        db.commit()
        db.refresh(user_profile)
        
        return ProfileResponse(
            message="Profile created successfully",
            session_id=preferences.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{session_id}", response_model=RecommendationResponse)
async def get_recommendations(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Generate job recommendations based on profile and Q&A"""
    try:
        user_profile = get_user_profile(db=db, session_id=session_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        search_service = get_search_service(db)
        search_service.retriever.strategy = SearchStrategy.HYBRID
        
        # Construct context query using both hard and soft constraints
        context_query = f"""
        Looking for jobs that match these skills: {', '.join(user_profile.skills)}
        with work culture preferences: {', '.join(user_profile.work_culture)}
        and values: {', '.join(user_profile.core_values)}
        Additional context from Q&A: {user_profile.additional_interests or ''}
        """
        
        results = search_service.search(
            query=context_query,
            db=db,
            session_id=session_id
        )
        
        # Process and rank results
        all_results = []
        seen_jobs = set()
        
        def normalize_text(text: str) -> str:
            return ' '.join(text.lower().split())

        def contains_phrase(text: str, phrase: str) -> bool:
            text = normalize_text(text)
            phrase = normalize_text(phrase)
            
            if phrase in text:
                return True
            
            words = phrase.split()
            if len(words) > 1:
                main_words = [w for w in words if len(w) > 3]
                for word in main_words:
                    if word not in text:
                        return False
                return True
            
            return False

        for job in results.get("jobs", []):
            if job["job_id"] not in seen_jobs:
                seen_jobs.add(job["job_id"])
                description = job.get("description", "").lower()
                
                matching_skills = [
                    skill for skill in user_profile.skills 
                    if contains_phrase(description, skill)
                ]
                
                matching_culture = [
                    culture for culture in user_profile.work_culture 
                    if contains_phrase(description, culture)
                ]
                
                skills_score = len(matching_skills) / len(user_profile.skills) if user_profile.skills else 0
                culture_score = len(matching_culture) / len(user_profile.work_culture) if user_profile.work_culture else 0
                
                match_score = (
                    skills_score * 0.6 +
                    culture_score * 0.4
                )

                if match_score > 0:
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
        
        all_results.sort(key=lambda x: x.match_score, reverse=True)
        top_recommendations = all_results[:10]
        
        return RecommendationResponse(
            recommendations=top_recommendations,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
def read_user(session_id: str, db: Session = Depends(get_db)):
    return get_user_profile(db=db, session_id=session_id)