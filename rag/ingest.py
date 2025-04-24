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

def setup_database():
    """Set up the PostgreSQL database with pgvector extension."""
    print("📊 Setting up database...")
    start_time = time.time()
    
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as connection:
        # Enable pgvector extension
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
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

if __name__ == "__main__":
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