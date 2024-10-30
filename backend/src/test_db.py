from database import SessionLocal
from sqlalchemy import text

def test_connection():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        print("Successfully connected to the database!")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()