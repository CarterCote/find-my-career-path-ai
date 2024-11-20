yo 

to run the backend, follow the instructions:

## Running the Server

1. Start the FastAPI server:
```bash
cd backend
uvicorn src.main:app --reload
```

2. In a separate terminal, start the Celery worker:
```bash
cd backend
celery -A src.celery_worker worker --loglevel=info
```

## Development

Make sure you have Redis running locally as it's required for Celery task queuing.