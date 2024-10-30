from sqlalchemy import Column, DateTime, Integer, String, Text, Boolean, JSON, func, ARRAY, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, unique=True)
    
    # Structured preferences
    core_values = Column(ARRAY(String))      # Top 3 core values
    work_culture = Column(ARRAY(String))     # Top 3 work culture preferences
    skills = Column(ARRAY(String))           # Top 3 skills
    top_six = Column(ARRAY(String))          # Top 6 overall preferences
    
    # Rankings/scores (optional)
    preference_rankings = Column(JSON)        # Store detailed rankings if needed
    
    # Free-form responses
    additional_interests = Column(Text)       # Free text for additional interests
    background = Column(Text)                # Previous experience, education, etc.
    goals = Column(Text)                     # Career goals
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    chat_messages = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    career_recommendations = relationship("CareerRecommendation", back_populates="user", cascade="all, delete-orphan")

class CareerRecommendation(Base):
    __tablename__ = "career_recommendations"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("user_profiles.session_id"), nullable=False)
    career_title = Column(String, nullable=False)
    reasoning = Column(Text, nullable=False)     # Why this career was recommended
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationship
    user = relationship("UserProfile", back_populates="career_recommendations")

class ChatHistory(Base):
    __tablename__ = "chat_histories"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("user_profiles.session_id"), nullable=False)
    message = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)    # True if user message, False if AI
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationship
    user = relationship("UserProfile", back_populates="chat_messages")
