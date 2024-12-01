from sqlalchemy.orm import Session
from . import models
from typing import Optional, List

def create_user_profile(db: Session, user_session_id: str):
    """Create or get a user profile"""
    user_profile = db.query(models.UserProfile).filter(
        models.UserProfile.user_session_id == user_session_id
    ).first()
    
    if not user_profile:
        user_profile = models.UserProfile(
            user_session_id=user_session_id,
            core_values=[],
            work_culture=[],
            skills=[],
            top_six=[]
        )
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)
    
    return user_profile

def get_user_profile(db: Session, user_session_id: str):
    """Get a user profile by user_session_id"""
    return db.query(models.UserProfile).filter(
        models.UserProfile.user_session_id == user_session_id
    ).first()

def create_chat_message(db: Session, chat_session_id: str, message: str, is_user: bool):
    """Create a chat message"""
    user_profile = get_user_profile(db, chat_session_id)
    if not user_profile:
        raise ValueError("User profile not found")
        
    chat_message = models.ChatHistory(
        user_id=user_profile.id,
        chat_session_id=chat_session_id,
        message=message,
        is_user=is_user
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    return chat_message

def create_job_recommendation(
    db: Session, 
    user_id: int, 
    chat_session_id: str,
    job_id: str,
    title: Optional[str] = None,
    company_name: Optional[str] = None,
    match_score: Optional[float] = None,
    recommendation_type: str = 'initial'
):
    """Create a job recommendation"""
    job_rec = models.JobRecommendation(
        user_id=user_id,
        chat_session_id=chat_session_id,
        job_id=job_id,
        title=title,
        company_name=company_name,
        match_score=match_score,
        recommendation_type=recommendation_type
    )
    db.add(job_rec)
    db.commit()
    db.refresh(job_rec)
    return job_rec

def get_user_job_recommendations(
    db: Session, 
    user_id: int, 
    limit: int = 10
) -> List[models.JobRecommendation]:
    """Get job recommendations for a user"""
    return db.query(models.JobRecommendation)\
        .filter(models.JobRecommendation.user_id == user_id)\
        .order_by(models.JobRecommendation.created_at.desc())\
        .limit(limit)\
        .all()

def get_session_job_recommendations(
    db: Session, 
    chat_session_id: str
) -> List[models.JobRecommendation]:
    """Get job recommendations for a specific chat session"""
    return db.query(models.JobRecommendation)\
        .filter(models.JobRecommendation.chat_session_id == chat_session_id)\
        .order_by(models.JobRecommendation.created_at.desc())\
        .all()