from fastapi import APIRouter, Depends, HTTPException, Query
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
    session_id: str = Query(default="default"),
    db: Session = Depends(get_db)
) -> Dict:
    """Conversational endpoint for interactive job search"""
    try:
        print(f"\nDebug - Chat endpoint called with session_id: {session_id}")
        search_service = get_search_service(db)
        
        # Store user message
        create_chat_message(db, session_id, message, is_user=True)
        print(f"Debug - Stored user message")
        
        try:
            # Get AI response using the chat method
            print(f"Debug - About to call search service chat")
            response = search_service.chat(
                message=message,
                session_id=session_id
            )
            print(f"Debug - Got response from search service: {response}")
            
            # Validate response format
            if not isinstance(response, dict) or 'response' not in response:
                response = {
                    'response': 'I apologize, but I need more information to help you effectively. Could you please provide more details about your career interests?',
                    'session_id': session_id
                }
                
        except Exception as chat_error:
            print(f"Chat service error: {str(chat_error)}")
            response = {
                'response': 'I encountered an issue processing your request. Could you please rephrase your question?',
                'session_id': session_id
            }
        
        # Store AI response
        create_chat_message(db, session_id, str(response.get('response')), is_user=False)
        print(f"Debug - Stored AI response")
        
        return response
        
    except Exception as e:
        print(f"Detailed error in chat endpoint: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your request. Please try again."
        )