from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel

class UserProfileBase(BaseModel):
    user_session_id: str
    core_values: Optional[List[str]] = None
    work_culture: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    top_six: Optional[List[str]] = None
    additional_interests: Optional[str] = None
    background: Optional[str] = None
    goals: Optional[str] = None
    preference_version: Optional[int] = 1

class UserProfileCreate(UserProfileBase):
    pass

class UserProfile(UserProfileBase):
    id: int
    created_at: datetime
    updated_at: datetime
    last_preference_update: datetime

    class Config:
        from_attributes = True

class ChatMessageBase(BaseModel):
    message: str
    is_user: bool

class ChatMessageCreate(ChatMessageBase):
    chat_session_id: str
    user_id: int

class ChatMessage(ChatMessageBase):
    id: int
    user_id: int
    chat_session_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class CareerRecommendationBase(BaseModel):
    career_title: str
    reasoning: str

class CareerRecommendationCreate(CareerRecommendationBase):
    user_session_id: str

class CareerRecommendation(CareerRecommendationBase):
    id: int
    user_session_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class JobRecommendationBase(BaseModel):
    job_id: str
    title: Optional[str] = None
    company_name: Optional[str] = None
    match_score: Optional[float] = None
    recommendation_type: Optional[str] = None
    matching_skills: Optional[List[str]] = []
    matching_culture: Optional[List[str]] = []
    evaluation_data: Optional[Dict] = None
    preference_version: Optional[int] = 1
    location: Optional[str] = None

class JobRecommendationCreate(JobRecommendationBase):
    user_id: int
    chat_session_id: str

class JobRecommendation(JobRecommendationBase):
    id: int
    user_id: int
    chat_session_id: str
    created_at: datetime

    class Config:
        from_attributes = True
