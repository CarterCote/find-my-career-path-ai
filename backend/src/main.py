from fastapi import Depends, FastAPI
from .utils.job_search import JobSearchService
from .utils.job_retriever import JobSearchRetriever
from .config import Settings
from sqlalchemy.orm import Session
from . import crud, models
from .database import get_db, SessionLocal
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from .crud import create_chat_message

app = FastAPI()
settings = Settings()

# Initialize embedding model
LlamaSettings.embed_model = OpenAIEmbedding()

# Initialize components with database session
db = SessionLocal()
retriever = JobSearchRetriever(db=db, embed_model=LlamaSettings.embed_model)
search_service = JobSearchService(retriever=retriever, settings=settings)

@app.post("/search/jobs")
async def search_jobs(query: str, session_id: str = "default"):
    """Direct job search endpoint for immediate results"""
    return search_service.search(query, session_id)

@app.post("/chat")
async def chat(message: str = "", session_id: str = "default", db: Session = Depends(get_db)):
    """Conversational endpoint for interactive job search"""
    try:
        # Validate API keys
        missing_keys = settings.validate_api_keys()
        if missing_keys:
            return {
                "response": f"Error: Missing required API keys: {', '.join(missing_keys)}. Please check your .env file.",
                "session_id": session_id
            }
        
        # If no message, return welcome message
        if not message:
            welcome_message = """Hello! I'm your AI career assistant. I can help you with:
            • Finding job opportunities that match your skills and interests
            • Career guidance and planning
            • Resume and interview tips
            • Salary insights and negotiation advice

            What would you like to know about?"""
            create_chat_message(db, session_id, welcome_message, is_user=False)
            return {"response": welcome_message, "session_id": session_id}
        
        # Process the chat message
        return search_service.search(message, db, session_id)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # For server logs
        return {
            "response": "I apologize, but I'm having trouble processing your request. Please try again.",
            "session_id": session_id,
            "error": str(e)
        }

@app.post("/users/")
def create_user(session_id: str, db: Session = Depends(get_db)):
    return crud.create_user_profile(db=db, session_id=session_id)

@app.get("/users/{session_id}")
def read_user(session_id: str, db: Session = Depends(get_db)):
    return crud.get_user_profile(db=db, session_id=session_id)