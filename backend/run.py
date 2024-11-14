import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    uvicorn.run("backend.src.main:app", host="127.0.0.1", port=8000, reload=True)