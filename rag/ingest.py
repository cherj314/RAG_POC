import os, sys, time, glob, re, concurrent.futures, traceback
from langchain.docstore.document import Document
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pdf_loader import PDFLoader
from semantic_text_splitter import SemanticTextSplitter

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  
SEMANTIC_SIMILARITY = float(os.getenv("SEMANTIC_SIMILARITY", "0.6"))
RESPECT_STRUCTURE = os.getenv("RESPECT_STRUCTURE", "true").lower()

DOCS_DIR = os.getenv("DOCS_DIR")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS"))

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Set up the PostgreSQL database with pgvector extension and required tables
def setup_database():
    print("📊 Setting up database...")
    start_time = time.time()
    
    engine = create_engine(CONNECTION_STRING)
    
    # Vector dimension operations and other database setup in a single transaction where possible
    operations = [
        # Enable pgvector extension
        "CREATE EXTENSION IF NOT EXISTS vector",
        
        # Handle collection table creation
        """
        CREATE TABLE IF NOT EXISTS langchain_pg_collection (
            uuid UUID PRIMARY KEY,
            name TEXT NOT NULL,
            cmetadata JSONB
        )
        """,
        
        # Drop existing index and embedding table to avoid dimension mismatch
        """
        DROP INDEX IF EXISTS langchain_pg_embedding_vector_idx;
        DROP TABLE IF EXISTS langchain_pg_embedding;
        """,
        
        # Create indexes once tables exist
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'langchain_pg_embedding'
            ) THEN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'langchain_pg_embedding_vector_idx'
                ) THEN
                    CREATE INDEX langchain_pg_embedding_vector_idx
                    ON langchain_pg_embedding
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'langchain_pg_embedding_collection_id_idx'
                ) THEN
                    CREATE INDEX langchain_pg_embedding_collection_id_idx
                    ON langchain_pg_embedding(collection_id);
                END IF;
                
                ANALYZE langchain_pg_embedding;
            END IF;
        END $$;
        """
    ]
    
    # Execute main operations in a transaction
    with engine.connect() as connection:
        for operation in operations:
            connection.execute(text(operation))
        connection.commit()
    
    # Set model name as session variable and optimize database parameters
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        try:
            connection.execute(text(f"SET app.embedding_model = '{EMBEDDING_MODEL}'"))
            connection.execute(text("ALTER SYSTEM SET work_mem = '32MB'"))
            connection.execute(text("SELECT pg_reload_conf()"))
        except Exception as e:
            print(f"  - Note: Some system settings could not be applied: {str(e)}")
    
    print(f"✅ Database setup completed in {time.time() - start_time:.2f}s")

# Create a semantic text splitter or fall back to recursive character splitter
def create_text_splitter(file_extension=""):
    print(f"  - Using semantic chunking (similarity threshold: {SEMANTIC_SIMILARITY})")
    return SemanticTextSplitter(
        embedding_model=EMBEDDING_MODEL,
        similarity_threshold=SEMANTIC_SIMILARITY,
        min_chunk_size=MIN_CHUNK_SIZE,
        max_chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        respect_structure=RESPECT_STRUCTURE,
        verbose=True
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

# Ensure text starts and ends with complete sentences
def ensure_complete_sentences(text):
    if not text or not text.strip():
        return text
        
    # Pattern to identify sentence boundaries and ending
    sentence_end_pattern = re.compile(r'[.!?][\'"]*\s+')
    sentence_end_final = re.compile(r'[.!?][\'"]*$')
    
    text = text.strip()
    
    # Check if text ends with a sentence terminator
    if not sentence_end_final.search(text):
        # Find the last sentence boundary
        match = list(sentence_end_pattern.finditer(text))
        if match:
            # Get the position of the last sentence terminator
            last_terminator_pos = match[-1].end()
            # Return the text up to that position
            text = text[:last_terminator_pos].strip()
    
    # Check if text starts with a capital letter
    # If not, it might be in the middle of a sentence
    if text and not text[0].isupper() and not text[0].isdigit() and not text[0] == '"':
        # Try to find the next sentence start
        match = re.search(r'[.!?][\'"]*\s+([A-Z0-9])', text)
        if match:
            # Start from the beginning of the next sentence
            text = text[match.start(1)-1:].strip()
    
    return text

# Process a document file, extract content, and split into chunks
def process_document(file_path):
    try:
        start_time = time.time()
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
        file_extension = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        
        print(f"\n📄 Processing {file_name} ({file_size:.1f} KB)...")
        
        # Load document based on file type
        if file_extension == '.pdf':
            # Configure PDF loader based on file size
            max_workers = min(2, os.cpu_count() or 1)
            batch_size = 10
            if file_size > 5120:  # 5 MB
                max_workers = min(4, os.cpu_count() or 1)
                batch_size = 15
            if file_size > 10240:  # 10 MB
                max_workers = min(8, os.cpu_count() or 1)
                batch_size = 20
                
            loader = PDFLoader(file_path, verbose=True, max_workers=max_workers, 
                              batch_size=batch_size, extract_images=False)
            document = loader.load()
            # Filter empty documents and ensure complete sentences
            document = [doc for doc in document if doc.page_content.strip()]
            for doc in document:
                doc.page_content = ensure_complete_sentences(doc.page_content)
        else:
            # For text files, read content and create a single document
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            document = [Document(
                page_content=ensure_complete_sentences(content),
                metadata={"source": file_path, "file_name": file_name, 
                         "file_id": file_id, "file_extension": file_extension}
            )]
        
        if not document:
            print(f"❌ No content extracted from {file_name}")
            return []
            
        print(f"  - Extracted {len(document)} document segments")
        
        # Update metadata for all documents
        for doc in document:
            doc.metadata.update({
                "file_name": file_name,
                "file_id": file_id,
                "file_extension": file_extension
            })
        
        # Split into chunks using semantic text splitter
        text_splitter = create_text_splitter(file_extension)
        chunks = text_splitter.split_documents(document)
        
        # Filter out header-only chunks
        header_patterns = [
            re.compile(r'^(?:CHAPTER|Chapter)\s+[\dIVXLC]+'),
            re.compile(r'^THE\s+[A-Z\s]+$'),
            re.compile(r'HARRY POTTER'),
            re.compile(r'^Page\s+\d+\s+of\s+\d+$'),
            re.compile(r'^\s*\d+\s*$')
        ]
        
        filtered_chunks = []
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk.page_content.strip()) < MIN_CHUNK_SIZE * 0.1:
                continue
                
            # Skip header-heavy chunks
            lines = chunk.page_content.strip().split('\n')
            header_lines = sum(1 for line in lines if any(p.search(line) for p in header_patterns))
            if lines and header_lines / len(lines) > 0.4:
                continue
                
            # Ensure complete sentences
            chunk.page_content = ensure_complete_sentences(chunk.page_content)
            filtered_chunks.append(chunk)
        
        chunks = filtered_chunks
        
        # For text files, check content coverage and add full document if needed
        if file_extension != '.pdf' and document[0].page_content:
            original_content = document[0].page_content
            chunks_content = " ".join(chunk.page_content for chunk in chunks)
            
            # Normalize whitespace for comparison
            norm_chunks = re.sub(r'\s+', ' ', chunks_content).strip()
            norm_original = re.sub(r'\s+', ' ', original_content).strip()
            
            # If coverage below 90%, add full document as additional chunk
            if norm_original and len(norm_chunks) / len(norm_original) < 0.9:
                print(f"  - Warning: Low content coverage, adding full document as chunk")
                chunks.append(Document(
                    page_content=ensure_complete_sentences(original_content),
                    metadata={
                        "source": file_path, "file_name": file_name, "file_id": file_id,
                        "file_extension": file_extension, "is_full_document": True
                    }
                ))
        
        # If no chunks created, use original documents
        if not chunks and document:
            print(f"  - Warning: No chunks created, using document segments")
            chunks = document
            
        print(f"✅ {file_name}: Created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        return chunks
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        return []

# Process multiple documents using optimized parallel processing
def process_documents(doc_files):
    if not doc_files:
        print("❌ No documents found for processing")
        return []
        
    print(f"🚀 Processing {len(doc_files)} documents...")
    start_time = time.time()
    all_chunks = []
    
    # Group files by type and size for optimized processing
    file_groups = {
        'text': sorted([f for f in doc_files if f.lower().endswith('.txt')], key=os.path.getsize),
        'pdf_small': [f for f in doc_files if f.lower().endswith('.pdf') and os.path.getsize(f) < 2 * 1024 * 1024],
        'pdf_medium': [f for f in doc_files if f.lower().endswith('.pdf') and 2 * 1024 * 1024 <= os.path.getsize(f) < 10 * 1024 * 1024],
        'pdf_large': [f for f in doc_files if f.lower().endswith('.pdf') and os.path.getsize(f) >= 10 * 1024 * 1024]
    }
    
    # Process file groups with appropriate parallelism
    processing_config = [
        ('text', file_groups['text'], MAX_WORKERS),
        ('small PDF', file_groups['pdf_small'], min(len(file_groups['pdf_small']), MAX_WORKERS // 2)),
        ('medium PDF', file_groups['pdf_medium'], 2),
    ]
    
    # Process text and PDF files with appropriate parallelism
    for group_name, files, max_workers in processing_config:
        if files:
            print(f"📄 Processing {len(files)} {group_name} files...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunks_list = list(executor.map(process_document, files))
                for chunks in chunks_list:
                    all_chunks.extend(chunks if chunks else [])
    
    # Process large PDFs sequentially to avoid memory issues
    if file_groups['pdf_large']:
        print(f"📄 Processing {len(file_groups['pdf_large'])} large PDF files sequentially...")
        for pdf_file in file_groups['pdf_large']:
            import gc
            gc.collect()  # Force garbage collection before processing large PDF
            
            print(f"  - Processing large PDF: {os.path.basename(pdf_file)} ({os.path.getsize(pdf_file) / (1024 * 1024):.1f} MB)")
            chunks = process_document(pdf_file)
            if chunks:
                all_chunks.extend(chunks)
            gc.collect()  # Force garbage collection after processing
    
    total_time = time.time() - start_time
    print(f"✅ Document processing {'completed' if all_chunks else 'failed'}: {'generated ' + str(len(all_chunks)) + ' chunks' if all_chunks else 'No chunks were generated'} in {total_time:.2f}s")
    
    return all_chunks

# Store chunks in the vector database with batching
def store_chunks_in_db(chunks, embeddings):
    if not chunks:
        print("No chunks to store.")
        return
        
    print(f"💾 Storing {len(chunks)} chunks in vector database...")
    start_time = time.time()
    
    # Ensure all chunks start and end with complete sentences
    for i, chunk in enumerate(chunks):
        chunks[i].page_content = ensure_complete_sentences(chunk.page_content)
    
    # Split chunks into batches and store each batch
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    
    for i, batch in enumerate(batches, 1):
        try:
            PGVector.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
                pre_delete_collection=(i == 1)  # Only delete on first batch
            )
            print(f"  ✅ Batch {i}/{len(batches)} stored")
        except Exception as e:
            print(f"  ❌ Error storing batch {i}: {e}")
    
    print(f"✅ All chunks stored in {time.time() - start_time:.2f}s")

# Main function to run the ingestion pipeline
def run_pipeline():
    print("\n" + "=" * 50 + "\n📚 Starting document ingestion pipeline\n" + "=" * 50 + "\n")
    pipeline_start = time.time()
    
    # Setup database and find documents
    setup_database()
    doc_files = find_documents()
    
    if not doc_files:
        print("❌ No documents found for processing. Add files to the Documents directory.")
        return
    
    # Process documents and generate chunks
    all_chunks = process_documents(doc_files)
    
    if not all_chunks:
        print("❌ No chunks were generated. Check your documents and processing settings.")
        return
    
    # Print content verification stats
    doc_counts = {}
    for chunk in all_chunks:
        source = chunk.metadata.get("file_name", "unknown")
        doc_counts[source] = doc_counts.get(source, 0) + 1
    
    print(f"\n📊 Content verification:\n  - Total documents: {len(doc_files)}\n  - Total chunks: {len(all_chunks)}")
    print("\n📑 Chunks by document:")
    for source, count in doc_counts.items():
        print(f"  - {source}: {count} chunks")
        
        # Check for emergency fallbacks in any chunk for this source
        if any(c.metadata.get("is_emergency_fallback") or 
               c.metadata.get("is_direct_file_fallback") or
               c.metadata.get("extraction_method") == "emergency_fallback" 
               for c in all_chunks if c.metadata.get("file_name") == source):
            print(f"    ⚠️ Used emergency fallback for this document")
    
    # Initialize embedding model and store chunks
    print(f"\n🧠 Initializing embedding model: {EMBEDDING_MODEL}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        store_chunks_in_db(all_chunks, embeddings)
    except Exception as e:
        print(f"❌ Error initializing embedding model: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        return
    
    print(f"\n⏱️ Total ingestion time: {time.time() - pipeline_start:.2f} seconds\n" + "=" * 50)

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