from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime
from ...database import get_db
from ...crud import create_user_profile, get_user_profile
from ...dependencies import get_search_service, get_settings
from ...utils.job_retriever import SearchStrategy
from ...models import UserProfile, JobRecommendation as DBJobRecommendation

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# Request/Response Models
class UserProfileResponse(BaseModel):
    id: int
    user_session_id: str
    core_values: List[str]
    work_culture: List[str]
    skills: List[str]
    additional_interests: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class JobRecommendationResponse(BaseModel):
    id: int
    user_session_id: str
    job_id: str
    title: str
    company_name: str
    match_score: float
    recommendation_type: str
    created_at: datetime

    class Config:
        from_attributes = True

class UserPreferences(BaseModel):
    user_session_id: str
    core_values: List[str]
    work_culture: List[str]
    skills: List[str]
    additional_interests: Optional[str] = None

class ProfileResponse(BaseModel):
    message: str
    user_session_id: str

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
    user_id: int
    recommendation_type: str
    preference_version: int

class RecommendationResponse(BaseModel):
    recommendations: List[JobRecommendation]
    user_session_id: str

# Endpoints - Reorder these routes
@router.post("/preferences", response_model=ProfileResponse)
async def create_user_preferences(
    preferences: UserPreferences,
    db: Session = Depends(get_db)
):
    """Create or update user profile with hard constraints"""
    try:
        user_profile = create_user_profile(
            db=db,
            user_session_id=preferences.user_session_id
        )
        
        user_profile.core_values = preferences.core_values
        user_profile.work_culture = preferences.work_culture
        user_profile.skills = preferences.skills
        user_profile.additional_interests = preferences.additional_interests
        
        db.commit()
        db.refresh(user_profile)
        
        return ProfileResponse(
            message="Profile created successfully",
            user_session_id=preferences.user_session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles", response_model=List[UserProfileResponse])
async def get_profiles(db: Session = Depends(get_db)):
    """Get all user profiles"""
    return db.query(UserProfile).all()

@router.get("/profile/{profile_id}/recommendations", response_model=List[JobRecommendationResponse])
async def get_profile_recommendations(
    profile_id: int,
    db: Session = Depends(get_db)
):
    """Get recommendations for a specific profile"""
    profile = db.query(UserProfile).filter(UserProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    recommendations = db.query(DBJobRecommendation)\
        .filter(DBJobRecommendation.user_session_id == profile.user_session_id)\
        .order_by(DBJobRecommendation.created_at.desc())\
        .all()
        
    return recommendations

@router.get("/recommendations/{user_session_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_session_id: str,
    db: Session = Depends(get_db)
):
    try:
        # Get user profile first
        user_profile = get_user_profile(db=db, user_session_id=user_session_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        # Get the search service instance
        search_service = get_search_service(db)
        
        # Check for stored recommendations
        stored_recs = db.query(DBJobRecommendation)\
            .filter(
                DBJobRecommendation.user_id == user_profile.id,
                DBJobRecommendation.chat_session_id == user_session_id
            )\
            .order_by(DBJobRecommendation.match_score.desc())\
            .all()
        
        if stored_recs:
            # Convert stored recommendations to response format
            recommendations = [JobRecommendation(
                job_id=rec.job_id,
                title=rec.title,
                company_name=rec.company_name,
                description="",  # We can add this if needed
                salary_range=None,  # We can add this if needed
                match_score=float(rec.match_score),
                matching_skills=rec.matching_skills or [],
                matching_culture=rec.matching_culture or [],
                location=rec.location,
                user_id=rec.user_id,
                recommendation_type=rec.recommendation_type,
                preference_version=rec.preference_version
            ) for rec in stored_recs]
            
            return RecommendationResponse(
                recommendations=recommendations,
                user_session_id=user_session_id
            )
        
        # If no stored recommendations, generate new ones
        context_query = f"Looking for jobs that match these skills: {', '.join(user_profile.skills or [])}"
        results = search_service.search(
            query=context_query,
            db=db,
            user_session_id=user_session_id
        )
        
        # Process and store new recommendations
        top_recommendations = results['jobs'][:5]
        search_service.store_recommendations(
            recommendations=top_recommendations,
            chat_session_id=user_session_id,
            user_id=user_profile.id,
            recommendation_type='initial'
        )
        
        return RecommendationResponse(
            recommendations=top_recommendations,
            user_session_id=user_session_id
        )
        
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/all/{user_session_id}", response_model=List[JobRecommendationResponse])
async def get_all_user_recommendations(
    user_session_id: str,
    db: Session = Depends(get_db)
):
    """Get all recommendations across chat sessions for a user"""
    user_profile = get_user_profile(db=db, user_session_id=user_session_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")
        
    # Get all recommendations for this user, grouped by chat session
    recommendations = db.query(DBJobRecommendation)\
        .filter(DBJobRecommendation.user_id == user_profile.id)\
        .order_by(
            DBJobRecommendation.chat_session_id,
            DBJobRecommendation.match_score.desc()
        )\
        .all()
    
    return recommendations

@router.get("/{user_session_id}")
def read_user(user_session_id: str, db: Session = Depends(get_db)):
    return get_user_profile(db=db, user_session_id=user_session_id)