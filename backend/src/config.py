from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    cohere_api_key: str
    langchain_api_key: str
    openai_api_key: str
    
    class Config:
        env_file = ".env"
    
    # def print_keys(self):
    #     """Print the status of all API keys."""
    #     for key, value in self.__dict__.items():
    #         if key.endswith('_api_key'):
    #             status = '✓ Set' if value and value.strip() else '✗ Missing or empty'
    #             print(f"{key}: {status}")