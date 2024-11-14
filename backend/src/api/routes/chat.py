from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict
from ...database import get_db
from ...crud import create_chat_message
from ...dependencies import get_search_service, get_settings

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("")
async def chat(
    message: str, 
    session_id: str = "default", 
    db: Session = Depends(get_db)
) -> Dict:
    """Conversational endpoint for interactive job search"""
    try:
        settings = get_settings()
        search_service = get_search_service(db)
        
        # Store user message
        create_chat_message(db, session_id, message, is_user=True)
        
        # Get AI response
        response = search_service.chat_engine.chat(
            message,
            session_id=session_id
        )
        
        # Store AI response
        create_chat_message(db, session_id, str(response), is_user=False)
        
        return {
            "response": str(response),
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))