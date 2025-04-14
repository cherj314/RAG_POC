import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.config import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

# Configuration
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
DOCS_DIR = "C:/Users/JohnChernoff/OneDrive - Arcurve/Desktop/RAGbot/Documents"  # Directory containing text files

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Create database tables
def setup_database():
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as connection:
        # Enable pgvector extension if not already enabled
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create the document_chunks table if it doesn't exist
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(1536)  -- Use the correct dimensionality for your embeddings
            );
        """))
        
        print("Database setup complete")

# Process a single document
def process_document(file_path):
    # Extract metadata from the file path
    file_name = os.path.basename(file_path)
    file_id = os.path.splitext(file_name)[0]
    
    # Load and parse the document
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()
    
    # Add metadata to the document
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
    
    print(f"Processed {file_name}: {len(chunks)} chunks created")
    return chunks

# Main pipeline function
def run_pipeline():
    # Setup database
    setup_database()
    
    # Process all documents in the directory
    all_chunks = []
    for file_name in os.listdir(DOCS_DIR):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DOCS_DIR, file_name)
            chunks = process_document(file_path)
            all_chunks.extend(chunks)
    
    print(f"Total chunks to be embedded: {len(all_chunks)}")
    
    # Store chunks in vector database
    try:
        # Initialize PGVector with our embeddings model and connection details
        db = PGVector.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=True  # Set to False if you want to append to existing collection
        )
        print(f"Successfully stored {len(all_chunks)} chunks with embeddings in the database")
    except Exception as e:
        print(f"Error storing vectors in database: {e}")

if __name__ == "__main__":
    run_pipeline()
