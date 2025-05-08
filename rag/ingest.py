import os, sys, time, glob, re, concurrent.futures, traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
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
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Chunking configuration
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE"))
SEMANTIC_SIMILARITY = float(os.getenv("SEMANTIC_SIMILARITY"))
RESPECT_STRUCTURE = os.getenv("RESPECT_STRUCTURE").lower()

# Processing parameters
DOCS_DIR = os.getenv("DOCS_DIR")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS"))

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Set up the PostgreSQL database with pgvector extension and optimizations
def setup_database():
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

# Create the appropriate text splitter based on configuration
def create_text_splitter(file_extension=""):
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

# Find all document files to be processed
def find_documents():
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

# Update the process_document function in ingest.py

def process_document(file_path):
    try:
        start_time = time.time()
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
        file_extension = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        
        print(f"\n📄 Processing {file_name} ({file_size:.1f} KB)...")
        
        # Store original file content for verification (only for text files)
        original_content = ""
        try:
            if file_extension.lower() != '.pdf':
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    original_content = f.read()
        except Exception as e:
            print(f"  - Note: Could not read original file for verification: {str(e)}")
        
        # Load the document based on file type with optimized settings
        if file_extension == '.pdf':
            # Use optimized settings based on file size
            if file_size > 10240:  # 10 MB
                # For large PDFs: more parallelism, larger batches
                max_workers = min(os.cpu_count() or 1, 8)  # Up to 8 workers
                batch_size = 20
            elif file_size > 5120:  # 5 MB
                max_workers = min(os.cpu_count() or 1, 4)  # Up to 4 workers
                batch_size = 15
            else:
                # Small PDFs: less parallelism to reduce overhead
                max_workers = 2
                batch_size = 10
                
            loader = PDFLoader(
                file_path, 
                verbose=True,
                max_workers=max_workers,
                batch_size=batch_size,
                extract_images=False  # Don't extract image info for better performance
            )
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            
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
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")

# Process multiple documents using optimized parallel processing
def process_documents(doc_files):
    all_chunks = []
    
    if not doc_files:
        print("❌ No documents found for processing")
        return all_chunks
        
    print(f"🚀 Processing {len(doc_files)} documents...")
    start_time = time.time()
    
    # Sort files by type and size for better processing
    text_files = sorted([f for f in doc_files if f.lower().endswith('.txt')], 
                        key=os.path.getsize)
    
    # For PDFs, sort by size but also partition into size groups for optimal processing
    pdf_files = sorted([f for f in doc_files if f.lower().endswith('.pdf')], 
                      key=os.path.getsize)
    
    # Separate PDFs into small, medium and large for optimized handling
    small_pdfs = [f for f in pdf_files if os.path.getsize(f) < 2 * 1024 * 1024]  # < 2MB
    medium_pdfs = [f for f in pdf_files if 2 * 1024 * 1024 <= os.path.getsize(f) < 10 * 1024 * 1024]  # 2-10MB
    large_pdfs = [f for f in pdf_files if os.path.getsize(f) >= 10 * 1024 * 1024]  # > 10MB
    
    # Process text files in parallel
    if text_files:
        print(f"📝 Processing {len(text_files)} text files...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunks_list = list(executor.map(process_document, text_files))
            for chunks in chunks_list:
                all_chunks.extend(chunks if chunks else [])
    
    # Process small PDFs in parallel (they're small enough to handle concurrently)
    if small_pdfs:
        print(f"📄 Processing {len(small_pdfs)} small PDF files in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(small_pdfs), MAX_WORKERS // 2)) as executor:
            chunks_list = list(executor.map(process_document, small_pdfs))
            for chunks in chunks_list:
                all_chunks.extend(chunks if chunks else [])
    
    # Process medium PDFs with limited parallelism to avoid memory issues
    if medium_pdfs:
        print(f"📄 Processing {len(medium_pdfs)} medium-sized PDF files with limited parallelism...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            chunks_list = list(executor.map(process_document, medium_pdfs))
            for chunks in chunks_list:
                all_chunks.extend(chunks if chunks else [])
    
    # Process large PDFs one at a time to avoid memory issues
    if large_pdfs:
        print(f"📄 Processing {len(large_pdfs)} large PDF files sequentially...")
        for pdf_file in large_pdfs:
            # Force garbage collection before processing large PDF
            import gc
            gc.collect()
            
            print(f"  - Processing large PDF: {os.path.basename(pdf_file)} ({os.path.getsize(pdf_file) / (1024 * 1024):.1f} MB)")
            chunks = process_document(pdf_file)
            if chunks:
                all_chunks.extend(chunks)
                
            # Force garbage collection after processing large PDF
            gc.collect()
    
    total_time = time.time() - start_time
    if all_chunks:
        print(f"✅ Document processing completed: generated {len(all_chunks)} chunks in {total_time:.2f}s")
    else:
        print("❌ No chunks were generated from documents")
    
    return all_chunks
        
# Store chunks in the vector database with batching
def store_chunks_in_db(chunks, embeddings):
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

# Main function to run the ingestion pipeline
def run_pipeline():
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