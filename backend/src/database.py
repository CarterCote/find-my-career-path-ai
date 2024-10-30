from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class JobPosting(Base):
    __tablename__ = 'job_postings'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True)
    company_name = Column(String)
    title = Column(String)
    description = Column(Text)
    processed_description = Column(Text)
    max_salary = Column(Float)
    min_salary = Column(Float)
    med_salary = Column(Float)
    location = Column(String)
    remote_allowed = Column(Boolean)
    job_category = Column(String)
    extracted_skills = Column(ARRAY(String))
    description_embedding = Column(ARRAY(Float))  # Store embeddings as array
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'company_name': self.company_name,
            'title': self.title,
            'description': self.description,
            'salary_range': f"${self.min_salary:,.0f} - ${self.max_salary:,.0f}" if self.min_salary and self.max_salary else "Not specified",
            'location': self.location,
            'remote_allowed': self.remote_allowed,
            'job_category': self.job_category,
            'skills': self.extracted_skills
        }