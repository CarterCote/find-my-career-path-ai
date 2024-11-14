from sqlalchemy.orm import Session
from . import models
from typing import Optional, List

def create_user_profile(db: Session, session_id: str):
    """Create or get a user profile"""
    user_profile = db.query(models.UserProfile).filter(
        models.UserProfile.session_id == session_id
    ).first()
    
    if not user_profile:
        user_profile = models.UserProfile(
            session_id=session_id,
            core_values=[],
            work_culture=[],
            skills=[],
            top_six=[]
        )
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)
    
    return user_profile

def get_user_profile(db: Session, session_id: str):
    """Get a user profile by session_id"""
    return db.query(models.UserProfile).filter(
        models.UserProfile.session_id == session_id
    ).first()

def create_chat_message(db: Session, session_id: str, message: str, is_user: bool):
    """Create a chat message"""
    chat_message = models.ChatHistory(
        session_id=session_id,
        message=message,
        is_user=is_user
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    return chat_message