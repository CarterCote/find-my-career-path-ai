from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ...database import get_db
from ..... import crud

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.post("/")
def create_user(session_id: str, db: Session = Depends(get_db)):
    return crud.create_user_profile(db=db, session_id=session_id)

@router.get("/{session_id}")
def read_user(session_id: str, db: Session = Depends(get_db)):
    return crud.get_user_profile(db=db, session_id=session_id)