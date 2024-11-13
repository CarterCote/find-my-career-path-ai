from database import SessionLocal
from sqlalchemy import text
import sys

def test_supabase_connection():
    try:
        # Create a session
        print("Attempting to connect to Supabase...")
        db = SessionLocal()
        
        # Test basic query
        result = db.execute(text("SELECT version()")).scalar()
        print(f"Connected successfully!")
        print(f"PostgreSQL version: {result}")
        
        # Test if pgvector extension is available
        vector_check = db.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'")).fetchone()
        print(f"pgvector extension is {'installed' if vector_check else 'NOT installed'}")
        
        # List all tables
        tables = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)).fetchall()
        print("\nAvailable tables:")
        for table in tables:
            print(f"- {table[0]}")
            
    except Exception as e:
        print(f"Connection failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    test_supabase_connection()