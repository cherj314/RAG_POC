import os
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool

# Load environment variables
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

DB_POOL = None

# Initialize the database connection pool
def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        try:
            DB_POOL = SimpleConnectionPool(
                2, 10,  # min, max connections
                dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
                host=DB_HOST, port=DB_PORT
            )
            print(f"Database pool initialized - connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except Exception as e:
            print(f"Error initializing database pool: {str(e)}")
            raise

# Get a connection from the pool
def get_db_connection():
    global DB_POOL
    if DB_POOL is None:
        init_db_pool()
    return DB_POOL.getconn()

# Release a connection back to the pool
def release_connection(conn):
    global DB_POOL
    if DB_POOL is not None:
        DB_POOL.putconn(conn)

# Get the UUID of the vector collection
def get_collection_id(conn):
    cur = conn.cursor()
    cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (COLLECTION_NAME,))
    result = cur.fetchone()
    cur.close()
    return result[0] if result else None