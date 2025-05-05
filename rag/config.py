import os
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "vectordb")
DB_USER = os.getenv("POSTGRES_USER", "myuser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypassword")

# Vector DB configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")

# Embedding Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Database connection pool
DB_POOL = None
DB_MIN_CONNECTIONS = 2
DB_MAX_CONNECTIONS = 10

def init_db_pool():
    """Initialize the database connection pool."""
    global DB_POOL
    if DB_POOL is None:
        try:
            DB_POOL = SimpleConnectionPool(
                DB_MIN_CONNECTIONS, DB_MAX_CONNECTIONS,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            print(f"Database pool initialized - connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except Exception as e:
            print(f"Error initializing database pool: {str(e)}")
            raise

def get_db_connection():
    """Get a connection from the connection pool."""
    global DB_POOL
    if DB_POOL is None:
        init_db_pool()
    return DB_POOL.getconn()

def release_connection(conn):
    """Return a connection to the connection pool."""
    global DB_POOL
    if DB_POOL is not None:
        DB_POOL.putconn(conn)

def get_collection_id(conn):
    """Get the UUID of the vector collection."""
    global COLLECTION_ID
    cur = conn.cursor()
    cur.execute("""
        SELECT uuid FROM langchain_pg_collection 
        WHERE name = %s;
    """, (COLLECTION_NAME,))
    result = cur.fetchone()
    cur.close()
    
    if result:
        return result[0]
    return None