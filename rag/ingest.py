import os
import sys
import time
import glob
import concurrent.futures
import traceback
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm
from pdf_loader import PDFLoader

# Import our custom semantic text splitter
try:
    from semantic_text_splitter import SemanticTextSplitter
except ImportError:
    print("⚠️ Could not import SemanticTextSplitter, falling back to RecursiveCharacterTextSplitter")
    SemanticTextSplitter = None

# Load environment variables
load_dotenv()

# Environment variables with defaults
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking configuration for both strategies
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
SEMANTIC_SIMILARITY = float(os.getenv("SEMANTIC_SIMILARITY", "0.75"))

DOCS_DIR = os.getenv("DOCS_DIR", "Documents")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def setup_database():
    """Set up the PostgreSQL database with pgvector extension and optimizations."""
    print("📊 Setting up database...")
    start_time = time.time()
    
    engine = create_engine(CONNECTION_STRING)
    
    # Split operations into regular operations (can be in transaction) and 
    # system operations (must be outside transaction)
    regular_operations = [
        # Enable pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector",
        
        # Update embedding table if it exists
        """
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
        """,
        
        # Create an IVFFlat index if needed
        """
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
        """,
        
        # Add an index on collection_id
        """
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
        """,
        
        # Analyze tables for query optimization
        """
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
        """
    ]
    
    system_operations = [
        # Optimize database parameters - must be outside transaction
        "ALTER SYSTEM SET work_mem = '32MB'",
        "SELECT pg_reload_conf()"
    ]
    
    # Execute regular operations in a transaction
    with engine.connect() as connection:
        for operation in regular_operations:
            connection.execute(text(operation))
        connection.commit()
    
    # Execute system operations outside of transactions
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        for operation in system_operations:
            connection.execute(text(operation))
    
    print(f"✅ Database setup completed in {time.time() - start_time:.2f}s")

def create_text_splitter(file_extension=""):
    """
    Create the appropriate text splitter based on configuration.
    
    Args:
        file_extension (str): File extension to adjust parameters if needed
        
    Returns:
        TextSplitter: A TextSplitter instance
    """
    # Try to use semantic chunking if selected
    if CHUNKING_STRATEGY == "semantic" and SemanticTextSplitter is not None:
        try:
            print(f"  - Using semantic chunking (similarity threshold: {SEMANTIC_SIMILARITY})")
            return SemanticTextSplitter(
                embedding_model=EMBEDDING_MODEL,
                similarity_threshold=SEMANTIC_SIMILARITY,
                min_chunk_size=MIN_CHUNK_SIZE,
                max_chunk_size=MAX_CHUNK_SIZE if file_extension != '.pdf' else MAX_CHUNK_SIZE + 200,
                chunk_overlap=CHUNK_OVERLAP,
                verbose=True
            )
        except Exception as e:
            print(f"⚠️ Error initializing semantic chunker: {str(e)}")
            print("  - Falling back to fixed-size chunking")
    
    # Fall back to fixed-size chunking
    chunk_size = CHUNK_SIZE + 200 if file_extension == '.pdf' else CHUNK_SIZE
    print(f"  - Using fixed-size chunking (size={chunk_size}, overlap={CHUNK_OVERLAP})")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ":", " ", ""]
    )

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
        loader = PDFLoader(file_path, verbose=True) if file_extension == '.pdf' else TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        
        if not document:
            print(f"❌ Failed to extract any content from {file_name}")
            return []
            
        print(f"  - Extracted {len(document)} document segments")
        
        # Add metadata to each document
        for doc in document:
            if not doc.metadata.get("file_name"):
                doc.metadata["file_name"] = file_name
            if not doc.metadata.get("file_id"):
                doc.metadata["file_id"] = file_id
        
        # Create text splitter (semantic or fixed-size)
        text_splitter = create_text_splitter(file_extension)
        
        # Split document into chunks
        try:
            chunks = text_splitter.split_documents(document)
            
            proc_time = time.time() - start_time
            print(f"✅ {file_name}: Created {len(chunks)} chunks in {proc_time:.2f}s")
            
            return chunks
        except Exception as e:
            print(f"❌ Error splitting document {file_name}: {str(e)}")
            print(f"Detailed error: {traceback.format_exc()}")
            
            # Fallback to basic chunking with even more basic splitter
            try:
                print("  - Trying fallback chunking method...")
                basic_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". "]
                )
                chunks = basic_splitter.split_documents(document)
                print(f"  - Fallback successful: Created {len(chunks)} chunks")
                return chunks
            except Exception as e2:
                print(f"❌ Fallback chunking also failed: {str(e2)}")
                return []
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        return []

def find_documents():
    """Find all document files to be processed"""
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Documents directory '{DOCS_DIR}' not found")
        return []
        
    # Find all supported files in the docs directory
    print(f"🔍 Searching for documents in '{DOCS_DIR}'...")
    txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    
    doc_files = txt_files + pdf_files
    print(f"  - Found {len(txt_files)} text files and {len(pdf_files)} PDF files")
    
    return doc_files

def process_documents(doc_files):
    """Process multiple documents using parallel processing when possible"""
    all_chunks = []
    
    if not doc_files:
        print("❌ No documents found for processing")
        return all_chunks
        
    print(f"🚀 Processing {len(doc_files)} documents...")
    start_time = time.time()
    
    # Sort files by type and size for better processing
    text_files = sorted([f for f in doc_files if f.lower().endswith('.txt')], 
                        key=os.path.getsize)
    pdf_files = sorted([f for f in doc_files if f.lower().endswith('.pdf')], 
                       key=os.path.getsize)
    
    # Process text files in parallel
    if text_files:
        print(f"📝 Processing {len(text_files)} text files...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunks_list = list(executor.map(process_document, text_files))
            for chunks in chunks_list:
                all_chunks.extend(chunks)
    
    # Process PDF files (one at a time to avoid memory issues)
    if pdf_files:
        print(f"📄 Processing {len(pdf_files)} PDF files...")
        for pdf_file in pdf_files:
            chunks = process_document(pdf_file)
            all_chunks.extend(chunks)
    
    total_time = time.time() - start_time
    if all_chunks:
        print(f"✅ Document processing completed: generated {len(all_chunks)} chunks in {total_time:.2f}s")
    else:
        print("❌ No chunks were generated from documents")
    
    return all_chunks

def store_chunks_in_db(chunks, embeddings):
    """Store chunks in the vector database using batching"""
    if not chunks:
        print("No chunks to store.")
        return
        
    print(f"💾 Storing {len(chunks)} chunks in vector database...")
    start_time = time.time()
    
    # Split chunks into batches
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    
    # Store each batch
    for i, batch in enumerate(batches, 1):
        batch_start = time.time()
        print(f"  - Storing batch {i}/{len(batches)} ({len(batch)} chunks)...")
        
        try:
            # Initialize PGVector with embeddings model and connection details
            PGVector.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
                pre_delete_collection=(i == 1)  # Only delete on first batch
            )
            
            batch_time = time.time() - batch_start
            print(f"    ✅ Batch {i}/{len(batches)} stored in {batch_time:.2f}s")
        except Exception as e:
            print(f"    ❌ Error storing batch {i}: {e}")
            print(f"    Detailed error: {traceback.format_exc()}")
    
    total_time = time.time() - start_time
    print(f"✅ All chunks stored in {total_time:.2f}s")

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
    
    # Process documents
    all_chunks = process_documents(doc_files)
    
    if not all_chunks:
        print("❌ No chunks were generated. Check your documents and processing settings.")
        return
    
    # Initialize embedding model
    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL}")
    model_start = time.time()
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"✅ Model initialized in {time.time() - model_start:.2f}s")
    except Exception as e:
        print(f"❌ Error initializing embedding model: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        return
    
    # Store chunks in vector database
    store_chunks_in_db(all_chunks, embeddings)
    
    total_time = time.time() - pipeline_start
    print(f"\n⏱️ Total ingestion time: {total_time:.2f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    # Add retry logic for container environments
    for attempt in range(5):
        try:
            run_pipeline()
            break
        except Exception as e:
            if "could not connect to server" in str(e).lower() and attempt < 4:
                retry_delay = 3 * (1.5 ** attempt)  # Exponential backoff
                print(f"Database connection failed. Retrying in {retry_delay:.1f}s... ({attempt+1}/5)")
                time.sleep(retry_delay)
            else:
                print(f"Fatal error: {e}")
                print(f"Detailed error: {traceback.format_exc()}")
                sys.exit(1)