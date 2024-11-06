from fastapi import FastAPI
from .api.routes import jobs, chat, users
from .config import Settings
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding

app = FastAPI()
settings = Settings()

# Initialize embedding model
LlamaSettings.embed_model = OpenAIEmbedding()

# Include routers
app.include_router(jobs.router)
app.include_router(chat.router)
app.include_router(users.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Job Search API"}