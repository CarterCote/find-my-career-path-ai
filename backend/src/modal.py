import modal
from fastapi import FastAPI
from src.main import app

# Create a stub for your application
stub = modal.Stub("pathways-api")

# Define your container image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "psycopg2-binary",
    "pgvector",
    "openai",
    "llama-index",
    "langchain",
    "python-dotenv",
    "pydantic",
    "pydantic-settings",
    # Add any other dependencies from your requirements.txt
)

# Create the ASGI app
@stub.function(
    image=image,
    secret=modal.Secret.from_dict({
        "OPENAI_API_KEY": "your_key_here",
        "SUPABASE_DB_HOST": "your_host",
        "SUPABASE_DB_PORT": "your_port",
        "SUPABASE_DB_NAME": "your_db_name",
        "SUPABASE_DB_USER": "your_user",
        "SUPABASE_DB_PASSWORD": "your_password",
        # Add other environment variables
    })
)
@modal.asgi_app()
def fastapi_app():
    return app