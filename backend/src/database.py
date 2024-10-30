from sqlalchemy import Column, Integer, String, Float, Boolean, BigInteger, Text, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://cartercote:3845@localhost/career_data"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# class JobPosting(Base):
#     __tablename__ = 'job_postings'
    
#     id = Column(Integer, primary_key=True)
#     job_id = Column(BigInteger, unique=True)
#     company_name = Column(Text)
#     title = Column(Text)
#     description = Column(Text)
#     max_salary = Column(Float)
#     pay_period = Column(String(50))
#     location = Column(Text)
#     company_id = Column(Float)
#     views = Column(Float)
#     med_salary = Column(Float, nullable=True)
#     min_salary = Column(Float)
#     formatted_work_type = Column(String(50))
#     applies = Column(Float)
#     original_listed_time = Column(BigInteger)
#     remote_allowed = Column(Boolean, nullable=True)
#     job_posting_url = Column(Text)
#     application_url = Column(Text, nullable=True)
#     application_type = Column(String(50))
#     expiry = Column(BigInteger)
#     closed_time = Column(BigInteger, nullable=True)
#     formatted_experience_level = Column(String(50), nullable=True)
#     skills_desc = Column(Text)
#     listed_time = Column(BigInteger)
#     posting_domain = Column(Text, nullable=True)
#     sponsored = Column(Boolean)
#     work_type = Column(String(50))
#     currency = Column(String(10))
#     compensation_type = Column(String(50))
#     normalized_salary = Column(Float)
#     zip_code = Column(String(10))
#     fips = Column(String(10))
#     processed_title = Column(Text)
#     processed_description = Column(Text)
#     processed_skills_desc = Column(Text)
#     job_category = Column(String(50))
#     processed_min_salary = Column(Float)
#     processed_med_salary = Column(Float, nullable=True)
#     processed_max_salary = Column(Float)
#     processed_location = Column(String(100))
#     processed_work_type = Column(String(50))
#     description_embedding = Column(Vector(768))  # For semantic search
    
#     def to_dict(self):
#         return {
#             'id': self.id,
#             'job_id': self.job_id,
#             'company_name': self.company_name,
#             'title': self.title,
#             'description': self.description,
#             'salary_range': f"${self.min_salary:,.0f} - ${self.max_salary:,.0f}" if self.min_salary and self.max_salary else "Not specified",
#             'location': self.location,
#             'remote_allowed': self.remote_allowed,
#             'job_category': self.job_category,
#         }