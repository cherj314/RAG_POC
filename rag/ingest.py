import os
import sys
import time
import glob
import concurrent.futures
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
from langchain.docstore.document import Document
import subprocess
import re
import shutil
import platform

# Import the PDFLoader class from pdf_loader.py module
from pdf_loader import PDFLoader

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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DOCS_DIR = os.getenv("DOCS_DIR", "Documents")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_env_file_if_needed():
    """Create or update the .env file with required variables if it doesn't exist."""
    if os.path.exists(".env"):
        print("✅ .env file already exists.")
        return
    
    print("🔧 Creating default .env file...")
    
    # Default values
    env_vars = {
        "POSTGRES_USER": "myuser",
        "POSTGRES_PASSWORD": "mypassword",
        "POSTGRES_DB": "vectordb",
        "DB_HOST": "postgres",  # Use 'postgres' for Docker networking
        "DB_PORT": "5432",
        "DB_NAME": "vectordb",
        "DB_USER": "myuser",
        "DB_PASSWORD": "mypassword",
        "API_PORT": "8000",
        "WEBUI_PORT": "3000",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "COLLECTION_NAME": "document_chunks",
        "CHUNK_SIZE": "600",
        "CHUNK_OVERLAP": "50",
        "WEBUI_AUTH_TOKEN": "default_token"
    }
    
    # Ask for OpenAI API key if not set
    env_vars["OPENAI_API_KEY"] = input("Enter your OpenAI API key (required): ").strip()
    if not env_vars["OPENAI_API_KEY"]:
        print("❌ No API key provided. You will need to set this in the .env file later.")
    
    # Write to .env file
    with open(".env", "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print("✅ Created new .env file")

def setup_documents_directory():
    """Create the Documents directory if it doesn't exist."""
    docs_dir = os.path.join(os.getcwd(), "Documents")
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"✅ Created Documents directory at {docs_dir}")
        
        # Create a sample document for testing
        sample_doc = os.path.join(docs_dir, "sample.txt")
        with open(sample_doc, "w") as f:
            f.write("This is a sample document for testing RAGbot's retrieval capabilities.\n\n")
            f.write("RAGbot uses retrieval-augmented generation to provide accurate and contextually relevant responses based on your documents.\n\n")
        print("✅ Created sample document")
    else:
        print(f"✅ Documents directory already exists at {docs_dir}")

def check_docker():
    """Check if Docker is running and properly configured."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not running. Please start Docker and try again.")
            return False
        
        print("✅ Docker is running")
        
        # Check for docker-compose
        compose_result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
        if compose_result.returncode != 0:
            print("❌ Docker Compose is not available. Please install Docker Compose.")
            return False
        
        print("✅ Docker Compose is available")
        return True
    
    except FileNotFoundError:
        print("❌ Docker is not installed. Please install Docker and try again.")
        return False

def create_helper_scripts():
    """Create helper scripts for easier system management."""
    # Determine if we're on Windows
    is_windows = platform.system() == "Windows"
    is_wsl = "microsoft-standard" in platform.uname().release.lower() if platform.system() == "Linux" else False
    
    if is_windows and not is_wsl:
        # Create batch files for Windows
        with open("ragbot-start.bat", "w") as f:
            f.write("""@echo off
echo Starting RAGbot...
docker compose up -d
echo.
echo Web interface will be available at http://localhost:3000
echo Use the default token from your .env file to log in
echo.
""")
        
        with open("ragbot-reset.bat", "w") as f:
            f.write("""@echo off
echo Stopping RAGbot and removing data...
docker compose down
docker volume rm ragbot_pgdata ragbot_openwebui-data
echo.
echo Starting fresh RAGbot instance...
docker compose up -d
echo.
echo Web interface will be available at http://localhost:3000
echo.
""")
        
        print("✅ Created Windows helper scripts: ragbot-start.bat and ragbot-reset.bat")
    
    else:
        # Create shell scripts for Linux/macOS/WSL
        with open("ragbot-start.sh", "w") as f:
            f.write("""#!/bin/bash
echo "Starting RAGbot..."
docker compose up -d
echo ""
echo "Web interface will be available at http://localhost:3000"
echo "Use the default token from your .env file to log in"
echo ""
""")
        
        with open("ragbot-reset.sh", "w") as f:
            f.write("""#!/bin/bash
echo "Stopping RAGbot and removing data..."
docker compose down
docker volume rm ragbot_pgdata ragbot_openwebui-data 2>/dev/null || true
echo ""
echo "Starting fresh RAGbot instance..."
docker compose up -d
echo ""
echo "Web interface will be available at http://localhost:3000"
echo ""
""")
        
        # Make scripts executable
        os.chmod("ragbot-start.sh", 0o755)
        os.chmod("ragbot-reset.sh", 0o755)
        
        print("✅ Created Unix helper scripts: ragbot-start.sh and ragbot-reset.sh")

def setup_database():
    """Set up the PostgreSQL database with pgvector extension and optimizations."""
    print("📊 Setting up database...")
    start_time = time.time()
    
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as connection:
        # Enable pgvector extension
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.commit()
        
        # Check if langchain_pg_embedding table exists and modify it
        connection.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'langchain_pg_embedding'
            ) THEN
                ALTER TABLE langchain_pg_embedding
                ALTER COLUMN embedding TYPE vector(384);
            END IF;
        END
        $$;
        """))
        connection.commit()
        
        # Create an IVFFlat index if it doesn't exist
        connection.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'langchain_pg_embedding'
            ) AND NOT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE indexname = 'langchain_pg_embedding_vector_idx'
            ) THEN
                CREATE INDEX langchain_pg_embedding_vector_idx
                ON langchain_pg_embedding
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            END IF;
        END
        $$;
        """))
        connection.commit()
        
        # Add an index on collection_id
        connection.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'langchain_pg_embedding'
            ) AND NOT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE indexname = 'langchain_pg_embedding_collection_id_idx'
            ) THEN
                CREATE INDEX langchain_pg_embedding_collection_id_idx
                ON langchain_pg_embedding(collection_id);
            END IF;
        END
        $$;
        """))
        connection.commit()
        
        # Analyze tables for query optimization
        connection.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'langchain_pg_embedding'
            ) THEN
                ANALYZE langchain_pg_embedding;
            END IF;
        END
        $$;
        """))
        connection.commit()
        
        # Increase work_mem for better performance
        connection.execute(text("ALTER SYSTEM SET work_mem = '32MB'"))
        connection.commit()
        
        # Reload configuration
        connection.execute(text("SELECT pg_reload_conf()"))
        connection.commit()
        
    print(f"✅ Database setup completed in {time.time() - start_time:.2f}s")

def process_document(file_path):
    """Process a single document file, extract metadata, and split into chunks."""
    try:
        start_time = time.time()
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
        file_extension = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        
        print(f"\n📄 Processing {file_name} ({file_size:.1f} KB)...")
        
        # Load the document based on file type
        if file_extension == '.pdf':
            # Use optimized PDF loader for PDF files
            loader = PDFLoader(file_path, verbose=True)
        else:
            # Default to text loader for other files
            print(f"  - Loading text file: {file_name}")
            loader = TextLoader(file_path, encoding="utf-8")
        
        # Load the document
        document = loader.load()
        
        if not document:
            print(f"❌ Failed to extract any content from {file_name}")
            return []
            
        print(f"  - Extracted {len(document)} document segments")
        
        # Add metadata to each document if not already present
        for doc in document:
            if not doc.metadata.get("file_name"):
                doc.metadata["file_name"] = file_name
            if not doc.metadata.get("file_id"):
                doc.metadata["file_id"] = file_id
        
        # Create a more efficient text splitter for PDFs
        separators = ["\n\n", "\n", ". ", "! ", "? ", ";", ":", " ", ""]
        
        # Adjust chunk size based on file type (larger for PDFs to reduce number of chunks)
        chunk_size = CHUNK_SIZE
        if file_extension == '.pdf':
            chunk_size = CHUNK_SIZE + 200  # Larger chunks for PDFs
        
        print(f"  - Splitting into chunks (size={chunk_size}, overlap={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=CHUNK_OVERLAP,
            separators=separators
        )
        
        # Split document into chunks
        chunks = text_splitter.split_documents(document)
        
        proc_time = time.time() - start_time
        print(f"✅ {file_name}: Created {len(chunks)} chunks in {proc_time:.2f}s")
        
        return chunks
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return []

def find_documents():
    """Find all document files to be processed"""
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Documents directory '{DOCS_DIR}' not found")
        return []
        
    # Find all supported files in the docs directory
    print(f"🔍 Searching for documents in '{DOCS_DIR}'...")
    doc_files = []
    txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    
    doc_files.extend(txt_files)
    doc_files.extend(pdf_files)
    
    print(f"  - Found {len(txt_files)} text files and {len(pdf_files)} PDF files")
    
    return doc_files

def batch_process_chunks(all_chunks, batch_size=BATCH_SIZE):
    """Process chunks in batches to avoid memory issues with large documents."""
    return [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]

def process_documents_in_parallel(doc_files):
    """Process multiple documents in parallel"""
    all_chunks = []
    
    if not doc_files:
        print("❌ No documents found for processing")
        return all_chunks
        
    print(f"🚀 Processing {len(doc_files)} documents in parallel...")
    start_time = time.time()
    
    # Process text files first, then PDFs (text files are usually faster)
    text_files = [f for f in doc_files if f.lower().endswith('.txt')]
    pdf_files = [f for f in doc_files if f.lower().endswith('.pdf')]
    
    # Sort files by size (smallest first for better parallelization)
    text_files.sort(key=lambda f: os.path.getsize(f))
    pdf_files.sort(key=lambda f: os.path.getsize(f))
    
    # Process text files
    if text_files:
        print(f"📝 Processing {len(text_files)} text files...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunks_list = list(executor.map(process_document, text_files))
            for chunks in chunks_list:
                all_chunks.extend(chunks)
    
    # Process PDF files
    if pdf_files:
        print(f"📄 Processing {len(pdf_files)} PDF files...")
        # Use fewer workers for PDFs as they're more resource intensive
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS // 2)) as executor:
            for pdf_file in pdf_files:
                # Process PDFs one at a time to avoid memory issues
                chunks = process_document(pdf_file)
                all_chunks.extend(chunks)
    
    total_time = time.time() - start_time
    if all_chunks:
        print(f"✅ Document processing completed: generated {len(all_chunks)} chunks in {total_time:.2f}s")
    else:
        print("❌ No chunks were generated from documents")
    
    return all_chunks

def store_chunks_in_db(chunks_batch, embeddings, batch_num, total_batches):
    """Store a batch of chunks in the vector database."""
    start_time = time.time()
    print(f"💾 Storing batch {batch_num}/{total_batches} ({len(chunks_batch)} chunks)...")
    
    try:
        # Initialize PGVector with embeddings model and connection details
        PGVector.from_documents(
            documents=chunks_batch,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=(batch_num == 1)  # Only delete on first batch
        )
        
        proc_time = time.time() - start_time
        print(f"✅ Batch {batch_num}/{total_batches} stored in {proc_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Error storing batch {batch_num}: {e}")
        return False

def run_pipeline():
    """Main ingestion pipeline with performance optimizations."""
    print("\n" + "=" * 50)
    print("📚 Starting optimized document ingestion pipeline")
    print("=" * 50 + "\n")
    
    pipeline_start = time.time()
    
    # Setup database
    setup_database()
    
    # Find all documents to process
    doc_files = find_documents()
    
    if not doc_files:
        print("❌ No documents found for processing. Add files to the Documents directory.")
        return
    
    # Process documents in parallel
    all_chunks = process_documents_in_parallel(doc_files)
    
    if not all_chunks:
        print("❌ No chunks were generated. Check your documents and processing settings.")
        return
    
    # Split chunks into batches to avoid memory issues
    chunks_batches = batch_process_chunks(all_chunks)
    total_batches = len(chunks_batches)
    
    # Initialize embedding model
    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL}")
    model_start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"✅ Model initialized in {time.time() - model_start:.2f}s")
    
    # Store chunks in vector database in batches
    print(f"💾 Storing {len(all_chunks)} chunks in {total_batches} batches...")
    
    success_count = 0
    for i, batch in enumerate(chunks_batches, 1):
        if store_chunks_in_db(batch, embeddings, i, total_batches):
            success_count += 1
    
    if success_count == total_batches:
        print(f"🎉 All {len(all_chunks)} chunks successfully stored with embeddings")
    else:
        print(f"⚠️ {success_count}/{total_batches} batches successfully stored")
    
    total_time = time.time() - pipeline_start
    print(f"\n⏱️ Total ingestion time: {total_time:.2f} seconds")
    print("=" * 50)

def setup():
    """Perform initial setup tasks."""
    print("\n" + "=" * 50)
    print("🤖 RAGbot Initial Setup")
    print("=" * 50)
    
    # Create .env file if it doesn't exist
    create_env_file_if_needed()
    
    # Set up Documents directory
    setup_documents_directory()
    
    # Check Docker installation
    if not check_docker():
        print("⚠️ Docker issues detected. Please fix Docker configuration before continuing.")
        return False
    
    # Create helper scripts
    create_helper_scripts()
    
    print("\n✅ RAGbot setup complete!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    # Check for setup argument
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup()
    else:
        # Add retry logic for container environments
        max_retries = 5
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                run_pipeline()
                break
            except Exception as e:
                if "could not connect to server" in str(e).lower() and attempt < max_retries - 1:
                    print(f"Database connection failed. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    print(f"Fatal error: {e}")
                    sys.exit(1)