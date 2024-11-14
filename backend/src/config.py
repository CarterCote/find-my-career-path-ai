from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str
    cohere_api_key: str
    langchain_api_key: str
    embed_model_name: str = "all-MiniLM-L6-v2"
    
    # Database settings
    supabase_db_host: str
    supabase_db_port: str
    supabase_db_name: str
    supabase_db_user: str
    supabase_db_password: str
    
    # LangChain settings
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_project: str = "find-my-career-path-ai"
    tokenizers_parallelism: bool = False

    class Config:
        env_file = ".env"
    
    # def print_keys(self):
    #     """Print the status of all API keys."""
    #     for key, value in self.__dict__.items():
    #         if key.endswith('_api_key'):
    #             status = '✓ Set' if value and value.strip() else '✗ Missing or empty'
    #             print(f"{key}: {status}")

@lru_cache()
def get_settings():
    return Settings()