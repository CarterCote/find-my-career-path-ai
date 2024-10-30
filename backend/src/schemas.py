from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class UserProfileBase(BaseModel):
    core_values: Optional[List[str]] = None
    work_culture: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    top_six: Optional[List[str]] = None
    additional_interests: Optional[str] = None
    background: Optional[str] = None
    goals: Optional[str] = None

class UserProfileCreate(UserProfileBase):
    session_id: str

class UserProfile(UserProfileBase):
    id: int
    session_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatMessageBase(BaseModel):
    message: str
    is_user: bool

class ChatMessageCreate(ChatMessageBase):
    session_id: str

class ChatMessage(ChatMessageBase):
    id: int
    session_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class CareerRecommendationBase(BaseModel):
    career_title: str
    reasoning: str

class CareerRecommendationCreate(CareerRecommendationBase):
    session_id: str

class CareerRecommendation(CareerRecommendationBase):
    id: int
    session_id: str
    created_at: datetime

    class Config:
        from_attributes = True
