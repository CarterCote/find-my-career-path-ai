from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, JSON, func, ARRAY, ForeignKey, BigInteger
from sqlalchemy.orm import relationship
from .database import Base
from pgvector.sqlalchemy import Vector

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True)
    user_session_id = Column(String, nullable=False)
    
    # Structured preferences
    core_values = Column(ARRAY(String))      # Top 10 core values
    work_culture = Column(ARRAY(String))     # Top 10 work culture preferences
    skills = Column(ARRAY(String))           # Top 10 skills
    top_six = Column(ARRAY(String))          # Top 6 overall preferences
    
    # Preference version tracking
    preference_version = Column(Integer, default=1)
    last_preference_update = Column(DateTime, server_default=func.now())
    
    # Free-form responses
    additional_interests = Column(Text)       # Free text for additional interests
    background = Column(Text)                # Previous experience, education, etc.
    goals = Column(Text)                     # Career goals
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    chat_messages = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    career_recommendations = relationship("CareerRecommendation", back_populates="user", cascade="all, delete-orphan")
    job_recommendations = relationship("JobRecommendation", back_populates="user", cascade="all, delete-orphan")
    
class CareerRecommendation(Base):
    __tablename__ = "career_recommendations"
    
    id = Column(Integer, primary_key=True)
    user_session_id = Column(String, ForeignKey("user_profiles.user_session_id"), nullable=False)
    career_title = Column(String(255), nullable=False)
    career_field = Column(String(255))
    reasoning = Column(Text, nullable=False)
    skills_required = Column(ARRAY(String))
    growth_potential = Column(Text)
    match_score = Column(Float)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationship
    user = relationship("UserProfile", back_populates="career_recommendations")
    example_jobs = relationship("JobRecommendation", secondary="career_job_examples")

class JobRecommendation(Base):
    __tablename__ = "job_recommendations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    chat_session_id = Column(String, nullable=False)
    preference_version = Column(Integer, nullable=False)  # Track which preference version generated this
    job_id = Column(String, ForeignKey("postings.job_id"), nullable=False)
    title = Column(String(255))
    company_name = Column(String(255))
    match_score = Column(Float)
    matching_skills = Column(ARRAY(String))
    matching_culture = Column(ARRAY(String))
    location = Column(String(255))
    recommendation_type = Column(String(50))
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("UserProfile", back_populates="job_recommendations")
    job = relationship("JobPosting", backref="recommendations")

# Association table to link careers with example jobs
class CareerJobExample(Base):
    __tablename__ = "career_job_examples"
    
    career_id = Column(Integer, ForeignKey("career_recommendations.id"), primary_key=True)
    job_id = Column(Integer, ForeignKey("job_recommendations.id"), primary_key=True)

class ChatHistory(Base):
    __tablename__ = "chat_histories"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    chat_session_id = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)    # True if user message, False if AI
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationship
    user = relationship("UserProfile", back_populates="chat_messages")

class JobPosting(Base):
    __tablename__ = "postings"
    
    job_id = Column(String(255), primary_key=True)
    company_name = Column(String(255))
    title = Column(String(255))
    description = Column(Text)
    max_salary = Column(Float(2))
    pay_period = Column(String(50))
    location = Column(String(255))
    company_id = Column(String(255))
    views = Column(Integer)
    med_salary = Column(Float(2))
    min_salary = Column(Float(2))
    description_embedding = Column(Vector(dim=384), nullable=True)
    structured_description = Column(JSON, nullable=True)
