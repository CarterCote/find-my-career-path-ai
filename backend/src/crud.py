from sqlalchemy import text
from sqlalchemy.orm import Session
from . import models, schemas

def create_user_profile(db: Session, session_id: str):
    db_user = models.UserProfile(session_id=session_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_profile(db: Session, session_id: str):
    return db.query(models.UserProfile).filter(models.UserProfile.session_id == session_id).first()

def create_chat_message(db: Session, session_id: str, message: str, is_user: bool):
    db_message = models.ChatHistory(
        session_id=session_id,
        message=message,
        is_user=is_user
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_chat_history(db: Session, session_id: str):
    return db.query(models.ChatHistory)\
        .filter(models.ChatHistory.session_id == session_id)\
        .order_by(models.ChatHistory.created_at.asc())\
        .all()


# def get_user(db: Session, user_id: int):
#     return db.query(models.User).filter(models.User.id == user_id).first()


# def get_user_by_phone_number(db: Session, phone_number: str):
#     return db.query(models.User).filter(models.User.phone_number == phone_number).first()


# def get_all_users(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.User).offset(skip).limit(limit).all()


# def create_user(db: Session, user: schemas.UserCreate):
#     hashed_password_fake = user.password + "notReallyHashed"
#     db_user = models.User(phone_number=user.phone_number,
#                           hashed_password=hashed_password_fake)
#     db.add(db_user)
#     db.commit()
#     db.refresh(db_user)
#     return db_user


# def get_all_entries(db: Session, skip: int = 0, limit: int = 100):
#     return db.query(models.Entry).offset(skip).limit(limit).all()


# def create_user_entry(db: Session, entry: schemas.EntryCreate, user_id: int):
#     db_entry = models.Entry(**entry.model_dump(), author_id=user_id)
#     db.add(db_entry)
#     db.commit()
#     db.refresh(db_entry)
#     return db_entry


# def query_embeddings(db: Session, user_id: int, embedding: list[float]):
#     sql = f"SELECT * FROM entries ORDER BY embedding <-> '{
#         embedding}' LIMIT 5;"
#     return db.execute(text(sql)).fetchall()
