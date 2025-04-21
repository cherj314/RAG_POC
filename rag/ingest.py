import os
import sys
import time
import glob
import concurrent.futures
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

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

# Maximum workers for parallel processing
MAX_WORKERS = 4

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def setup_database():
    """
    Set up the PostgreSQL database with pgvector extension and required tables.
    """
    print("🔄 Setting up database...")
    start_time = time.time()
    
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
        
        connection.commit()
        
    print(f"✅ Database setup complete in {time.time() - start_time:.2f}s")

def process_document(file_path):
    """
    Process a single document file, extract metadata, and split into chunks.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        list: List of document chunks with metadata
    """
    try:
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
        
        return chunks
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return []

def find_documents():
    """Find all document files to be processed"""
    if not os.path.exists(DOCS_DIR):
        print(f"❌ Error: Documents directory '{DOCS_DIR}' not found")
        return []
        
    # Find all text files in the docs directory
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    
    # Add support for more file types if needed
    # doc_files += glob.glob(os.path.join(DOCS_DIR, "*.md"))
    # doc_files += glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    
    return doc_files

def process_documents_in_parallel(doc_files):
    """Process multiple documents in parallel"""
    all_chunks = []
    
    if not doc_files:
        print("❌ No documents found for processing")
        return all_chunks
        
    print(f"🔄 Processing {len(doc_files)} documents in parallel...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit processing jobs and create a future-to-filename mapping
        future_to_file = {executor.submit(process_document, file_path): file_path 
                        for file_path in doc_files}
        
        # Process as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(doc_files), 
                          desc="Processing documents"):
            file_path = future_to_file[future]
            try:
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)
                    print(f"  ✅ {os.path.basename(file_path)}: {len(chunks)} chunks")
                else:
                    print(f"  ⚠️ {os.path.basename(file_path)}: No chunks extracted")
            except Exception as e:
                print(f"  ❌ {os.path.basename(file_path)}: Error - {str(e)}")
    
    print(f"✅ Document processing complete in {time.time() - start_time:.2f}s")
    return all_chunks

def run_pipeline():
    """
    Main ingestion pipeline: setup database, process documents,
    generate embeddings, and store in vector database.
    """
    total_start_time = time.time()
    print("🚀 Starting document ingestion pipeline")
    
    # Setup database
    setup_database()
    
    # Find all documents to process
    doc_files = find_documents()
    
    # Process documents in parallel
    all_chunks = process_documents_in_parallel(doc_files)
    
    if not all_chunks:
        print("⚠️ No chunks were generated. Check your documents and processing settings.")
        return
        
    # Initialize embedding model
    print(f"🔄 Initializing embedding model: {EMBEDDING_MODEL}")
    embedding_start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"✅ Embedding model loaded in {time.time() - embedding_start:.2f}s")
    
    # Store chunks in vector database
    print(f"🔄 Storing {len(all_chunks)} chunks in vector database...")
    db_start_time = time.time()
    
    try:
        # Initialize PGVector with embeddings model and connection details
        db = PGVector.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=True  # Set to False to append instead of replace
        )
        
        db_time = time.time() - db_start_time
        total_time = time.time() - total_start_time
        
        print(f"✅ Successfully stored {len(all_chunks)} chunks with embeddings in {db_time:.2f}s")
        print(f"✅ Total ingestion pipeline completed in {total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error storing vectors in database: {e}")

if __name__ == "__main__":
    # Add retry logic for container environments where the database might not be ready
    max_retries = 5
    retry_delay = 3  # seconds
    
    for attempt in range(max_retries):
        try:
            run_pipeline()
            break
        except Exception as e:
            if "could not connect to server" in str(e).lower() and attempt < max_retries - 1:
                print(f"⚠️ Database connection failed. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"❌ Fatal error: {e}")
                sys.exit(1)