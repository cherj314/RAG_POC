from psycopg2.pool import SimpleConnectionPool

# Database
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "vectordb"
DB_USER = "myuser"
DB_PASSWORD = "mypassword"

# Vector DB
COLLECTION_NAME = "document_chunks"
COLLECTION_ID = None

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Initialize pool at startup
DB_POOL = None

def get_db_connection():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = SimpleConnectionPool(
            1, 10,  # min, max connections
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    return DB_POOL.getconn()

def release_connection(conn):
    global DB_POOL
    if DB_POOL is not None:
        DB_POOL.putconn(conn)

def get_collection_id(conn):
    global COLLECTION_ID
    if COLLECTION_ID is None:
        cur = conn.cursor()
        cur.execute("""
            SELECT uuid FROM langchain_pg_collection 
            WHERE name = %s;
        """, (COLLECTION_NAME,))
        COLLECTION_ID = cur.fetchone()[0]
        cur.close()
    return COLLECTION_ID