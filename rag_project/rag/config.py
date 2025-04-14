# rag/config.py

# Database
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "vectordb"
DB_USER = "myuser"
DB_PASSWORD = "mypassword"

# Vector DB
COLLECTION_NAME = "document_chunks"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
