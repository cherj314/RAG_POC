import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration from .env
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DOCS_DIR = os.getenv("DOCS_DIR", "Documents")

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def setup_database():
    """
    Set up the PostgreSQL database with pgvector extension and required tables.
    """
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as connection:
        # Enable pgvector extension
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create document_chunks table if needed
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(384)  -- Dimensionality based on the embedding model
            );
        """))
        
        print("✅ Database setup complete")

def process_document(file_path):
    """
    Process a single document file, extract metadata, and split into chunks.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        list: List of document chunks with metadata
    """
    # Extract metadata from file path
    file_name = os.path.basename(file_path)
    file_id = os.path.splitext(file_name)[0]
    
    # Load the document
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()
    
    # Add metadata to each document
    for doc in document:
        doc.metadata["file_name"] = file_name
        doc.metadata["file_id"] = file_id
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(document)
    
    print(f"📄 Processed {file_name}: {len(chunks)} chunks created")
    return chunks

def run_pipeline():
    """
    Main ingestion pipeline: setup database, process documents,
    generate embeddings, and store in vector database.
    """
    # Setup database
    setup_database()
    
    # Initialize embedding model
    print(f"🔄 Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Process all text documents in the directory
    all_chunks = []
    print(f"📁 Scanning directory: {DOCS_DIR}")
    
    if not os.path.exists(DOCS_DIR):
        print(f"❌ Error: Documents directory '{DOCS_DIR}' not found")
        return
        
    for file_name in os.listdir(DOCS_DIR):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DOCS_DIR, file_name)
            chunks = process_document(file_path)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        print("❌ No text documents found for processing")
        return
        
    print(f"🔢 Total chunks to be embedded: {len(all_chunks)}")
    
    # Store chunks in vector database
    try:
        # Initialize PGVector with embeddings model and connection details
        db = PGVector.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=True  # Set to False to append instead of replace
        )
        print(f"✅ Successfully stored {len(all_chunks)} chunks with embeddings in the database")
    except Exception as e:
        print(f"❌ Error storing vectors in database: {e}")

if __name__ == "__main__":
    run_pipeline()