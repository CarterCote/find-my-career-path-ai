from sqlalchemy import text
from sqlalchemy.orm import Session

def query_job_embeddings(db: Session, embedding: list[float], limit: int = 5):
    embedding_string = ','.join(map(str, embedding))
    sql = text(f"""
        SELECT 
            id, 
            title,
            company_name,
            processed_location,
            processed_description,
            processed_min_salary,
            processed_max_salary,
            remote_allowed,
            job_category,
            1 - (description_embedding <=> ARRAY[{embedding_string}]::vector) AS similarity_score
        FROM job_postings
        ORDER BY description_embedding <=> ARRAY[{embedding_string}]::vector
        LIMIT :limit
    """)
    return db.execute(sql, {"limit": limit}).fetchall()