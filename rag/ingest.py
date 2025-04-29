import os
import sys
import time
import glob
import re
import concurrent.futures
import traceback
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
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
DB_NAME = os.getenv("POSTGRES_DB", "vectordb")
DB_USER = os.getenv("POSTGRES_USER", "myuser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking configuration
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
SEMANTIC_SIMILARITY = float(os.getenv("SEMANTIC_SIMILARITY", "0.75"))
RESPECT_STRUCTURE = os.getenv("RESPECT_STRUCTURE", "true").lower() == "true"

# Processing parameters
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
    
    # First handle vector dimension update for the new embedding model
    # This needs to happen before other operations
    vector_dim_operations = [
    # Drop existing index first if it exists
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE indexname = 'langchain_pg_embedding_vector_idx'
        ) THEN
            DROP INDEX langchain_pg_embedding_vector_idx;
        END IF;
    END
    $$;
    """,
    
    # Drop the table if it exists to avoid dimension mismatch
    """
    DROP TABLE IF EXISTS langchain_pg_embedding;
    """,
    
    # Create the table with the correct dimension based on model
    """
    DO $$
    DECLARE
        vector_dim INT;
    BEGIN
        -- Set vector dimension based on model name
        IF '{}' LIKE '%mpnet%' THEN
            vector_dim := 768;
        ELSE
            vector_dim := 384;
        END IF;
        
        -- Log the dimension being used
        RAISE NOTICE 'Setting vector dimension to %', vector_dim;
        
        -- Create the collection table if needed
        IF NOT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'langchain_pg_collection'
        ) THEN
            CREATE TABLE langchain_pg_collection (
                uuid UUID PRIMARY KEY,
                name TEXT NOT NULL,
                cmetadata JSONB
            );
        END IF;
    END;
    $$;
    """.format(EMBEDDING_MODEL)  # Use format to insert the model name directly
    ]
    
    # Execute the vector dimension update operations
    for operation in vector_dim_operations:
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
                connection.execute(text(operation))
                print(f"  - Executed vector dimension update")
        except Exception as e:
            print(f"  - Warning: Vector dimension update operation failed: {str(e)}")
    
    # Try to set the model name as a session variable
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text(f"SET app.embedding_model = '{EMBEDDING_MODEL}'"))
    except:
        print("  - Note: Could not set model name as session variable")
    
    # Regular database operations
    regular_operations = [
        # Enable pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector",
        
        # Create an IVFFlat index if needed (with the right dimensions)
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
                respect_structure=RESPECT_STRUCTURE,
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
        
        # Store original file content for verification
        original_content = ""
        try:
            if file_extension.lower() != '.pdf':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    original_content = f.read()
        except Exception as e:
            print(f"  - Note: Could not read original file for verification: {str(e)}")
        
        # Load the document based on file type
        loader = PDFLoader(file_path, verbose=True) if file_extension == '.pdf' else TextLoader(file_path, encoding="utf-8")
        document = loader.load()
        
        if not document:
            print(f"❌ Failed to extract any content from {file_name}")
            
            # Last resort for non-PDF files: create a single document with the whole file
            if original_content and file_extension.lower() != '.pdf':
                print(f"  - Creating emergency document with entire file content")
                document = [Document(
                    page_content=original_content,
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "file_id": file_id,
                        "file_extension": file_extension,
                        "extraction_method": "emergency_fallback"
                    }
                )]
            else:
                return []
            
        print(f"  - Extracted {len(document)} document segments")
        
        # Add metadata to each document
        for doc in document:
            if not doc.metadata.get("file_name"):
                doc.metadata["file_name"] = file_name
            if not doc.metadata.get("file_id"):
                doc.metadata["file_id"] = file_id
            doc.metadata["file_extension"] = file_extension
        
        # Create text splitter (enhanced semantic or fixed-size)
        text_splitter = create_text_splitter(file_extension)
        
        # Split document into chunks
        try:
            chunks = text_splitter.split_documents(document)
            
            # Verify chunk content covers the entire original document
            if chunks:
                # Check chunk coverage for text files (PDFs are harder to verify)
                if original_content and file_extension.lower() != '.pdf':
                    total_chunks_content = " ".join([chunk.page_content for chunk in chunks])
                    # Normalize whitespace for comparison
                    chunks_content = re.sub(r'\s+', ' ', total_chunks_content).strip()
                    original_normalized = re.sub(r'\s+', ' ', original_content).strip()
                    
                    # Calculate content coverage
                    coverage_ratio = len(chunks_content) / len(original_normalized) if len(original_normalized) > 0 else 0
                    print(f"  - Content coverage: {coverage_ratio:.2f} ({len(chunks_content)} / {len(original_normalized)} chars)")
                    
                    # If we've lost significant content, fall back to a single chunk with the entire document
                    if coverage_ratio < 0.9 and len(original_normalized) > 0:
                        print(f"  - Warning: Content coverage below 90%, adding full document as an additional chunk")
                        chunks.append(Document(
                            page_content=original_content,
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_extension": file_extension,
                                "chunk": len(chunks),
                                "total_chunks": len(chunks) + 1,
                                "is_full_document": True
                            }
                        ))
            
            proc_time = time.time() - start_time
            print(f"✅ {file_name}: Created {len(chunks)} chunks in {proc_time:.2f}s")
            
            # If we somehow ended up with no chunks, use the original documents as chunks
            if not chunks and document:
                print(f"  - Warning: No chunks created, using document segments as chunks")
                chunks = document
            
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
                
                # If chunks were created but we still suspect content loss, add full document
                if chunks and original_content and file_extension.lower() != '.pdf':
                    # Check content coverage
                    total_chunks_content = " ".join([chunk.page_content for chunk in chunks])
                    chunks_content = re.sub(r'\s+', ' ', total_chunks_content).strip()
                    original_normalized = re.sub(r'\s+', ' ', original_content).strip()
                    
                    # Calculate content coverage
                    coverage_ratio = len(chunks_content) / len(original_normalized) if len(original_normalized) > 0 else 0
                    print(f"  - Fallback content coverage: {coverage_ratio:.2f} ({len(chunks_content)} / {len(original_normalized)} chars)")
                    
                    # Add full document if coverage is low
                    if coverage_ratio < 0.9:
                        print(f"  - Adding full document as a fallback chunk")
                        chunks.append(Document(
                            page_content=original_content,
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_extension": file_extension,
                                "chunk": len(chunks),
                                "total_chunks": len(chunks) + 1,
                                "is_full_document": True,
                                "is_fallback": True
                            }
                        ))
                
                return chunks
            except Exception as e2:
                print(f"❌ Fallback chunking also failed: {str(e2)}")
                
                # Ultimate fallback: just return the original documents
                if document:
                    print(f"  - Emergency fallback: Using original document segments as chunks")
                    # Don't return the raw documents, this ensures they have proper metadata
                    emergency_chunks = []
                    for i, doc in enumerate(document):
                        emergency_chunks.append(Document(
                            page_content=doc.page_content,
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_extension": file_extension,
                                "chunk": i,
                                "total_chunks": len(document),
                                "is_emergency_fallback": True,
                                # Include any existing metadata
                                **doc.metadata
                            }
                        ))
                    return emergency_chunks
                else:
                    # If all else fails and we have the original content, create a single document
                    if original_content:
                        print(f"  - Last resort fallback: Creating a single chunk with entire document content")
                        return [Document(
                            page_content=original_content,
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_extension": file_extension,
                                "is_emergency_fallback": True
                            }
                        )]
                    return []
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        
        # Absolute last resort - try to read the file directly
        try:
            if file_extension.lower() != '.pdf':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    if content.strip():
                        print(f"  - Critical error recovery: Creating document from direct file read")
                        return [Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_extension": file_extension,
                                "is_direct_file_fallback": True
                            }
                        )]
        except Exception as e2:
            print(f"  - Final recovery attempt failed: {str(e2)}")
        
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
    """Main ingestion pipeline with performance optimizations and content verification."""
    print("\n" + "=" * 50)
    print("📚 Starting document ingestion pipeline")
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
    
    # Content verification stats
    print("\n📊 Content verification:")
    print(f"  - Total documents processed: {len(doc_files)}")
    print(f"  - Total chunks generated: {len(all_chunks)}")
    
    # Analyze chunks by document source
    docs_by_source = {}
    for chunk in all_chunks:
        source = chunk.metadata.get("file_name", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(chunk)
    
    print("\n📑 Chunks by document:")
    for source, chunks in docs_by_source.items():
        print(f"  - {source}: {len(chunks)} chunks")
        
        # Check for emergency fallbacks
        emergency_chunks = [c for c in chunks if c.metadata.get("is_emergency_fallback") or 
                           c.metadata.get("is_direct_file_fallback") or
                           c.metadata.get("extraction_method") == "emergency_fallback"]
        if emergency_chunks:
            print(f"    ⚠️ Used emergency fallback for this document")
    
    # Initialize embedding model
    print(f"\n🧠 Initializing embedding model: {EMBEDDING_MODEL}")
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