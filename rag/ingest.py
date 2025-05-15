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

# Get environment variables with defaults
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
RESPECT_STRUCTURE = os.getenv("RESPECT_STRUCTURE", "true").lower() == "true"
DOCS_DIR = os.getenv("DOCS_DIR")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS"))

# Database connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def setup_database():
    """Set up PostgreSQL database with pgvector extension"""
    print("📊 Setting up database...")
    start_time = time.time()
    
    engine = create_engine(CONNECTION_STRING)
    operations = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS langchain_pg_collection (
            uuid UUID PRIMARY KEY, name TEXT NOT NULL, cmetadata JSONB)""",
        """DROP INDEX IF EXISTS langchain_pg_embedding_vector_idx;
           DROP TABLE IF EXISTS langchain_pg_embedding;""",
        """DO $$
        BEGIN
            IF EXISTS (SELECT FROM information_schema.tables 
                      WHERE table_schema = 'public' AND table_name = 'langchain_pg_embedding') THEN
                IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'langchain_pg_embedding_vector_idx') THEN
                    CREATE INDEX langchain_pg_embedding_vector_idx
                    ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                END IF;
                
                IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'langchain_pg_embedding_collection_id_idx') THEN
                    CREATE INDEX langchain_pg_embedding_collection_id_idx ON langchain_pg_embedding(collection_id);
                END IF;
                ANALYZE langchain_pg_embedding;
            END IF;
        END $$;"""
    ]
    
    # Execute operations and optimize DB parameters
    with engine.connect() as connection:
        for operation in operations:
            connection.execute(text(operation))
        connection.commit()
    
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        try:
            connection.execute(text(f"SET app.embedding_model = '{EMBEDDING_MODEL}'"))
            connection.execute(text("ALTER SYSTEM SET work_mem = '32MB'"))
            connection.execute(text("SELECT pg_reload_conf()"))
        except Exception as e:
            print(f"  - Note: Some system settings could not be applied: {str(e)}")
    
    print(f"✅ Database setup completed in {time.time() - start_time:.2f}s")

def ensure_complete_sentences(text):
    """Ensure text starts and ends with complete sentences"""
    if not text or not text.strip():
        return text
        
    sentence_end_pattern = re.compile(r'[.!?][\'"]*\s+')
    sentence_end_final = re.compile(r'[.!?][\'"]*$')
    
    text = text.strip()
    
    # Check if text ends with a sentence terminator
    if not sentence_end_final.search(text):
        # Find the last sentence boundary
        match = list(sentence_end_pattern.finditer(text))
        if match:
            text = text[:match[-1].end()].strip()
    
    # Check if text starts with a capital letter
    if text and not text[0].isupper() and not text[0].isdigit() and not text[0] == '"':
        match = re.search(r'[.!?][\'"]*\s+([A-Z0-9])', text)
        if match:
            text = text[match.start(1)-1:].strip()
    
    return text

def process_document(file_path):
    """Process a document file, extract content, and split into chunks"""
    try:
        start_time = time.time()
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
        file_extension = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        
        print(f"\n📄 Processing {file_name} ({file_size:.1f} KB)...")
        
        # Load document based on file type
        if file_extension == '.pdf':
            loader = PDFLoader(file_path, verbose=True, extract_images=False)
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
        splitter = SemanticTextSplitter(
            embedding_model=EMBEDDING_MODEL,
            similarity_threshold=SEMANTIC_SIMILARITY,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            respect_structure=RESPECT_STRUCTURE,
            verbose=True
        )
        chunks = splitter.split_documents(document)
        
        # Filter out very short chunks
        filtered_chunks = []
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk.page_content.strip()) < MIN_CHUNK_SIZE * 0.1:
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

def process_documents(doc_files):
    """Process multiple documents in parallel"""
    if not doc_files:
        print("❌ No documents found for processing")
        return []
        
    print(f"🚀 Processing {len(doc_files)} documents...")
    start_time = time.time()
    all_chunks = []
    
    # Process all files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunks_list = list(executor.map(process_document, doc_files))
        for chunks in chunks_list:
            all_chunks.extend(chunks if chunks else [])
    
    print(f"✅ Document processing: generated {len(all_chunks)} chunks in {time.time() - start_time:.2f}s")
    return all_chunks

def store_chunks_in_db(chunks, embeddings):
    """Store chunks in the vector database with batching"""
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

def run_pipeline():
    """Main function to run the ingestion pipeline"""
    print("\n" + "=" * 50 + "\n📚 Starting document ingestion pipeline\n" + "=" * 50 + "\n")
    pipeline_start = time.time()
    
    # Setup database and find documents
    setup_database()
    
    # Find document files
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Documents directory '{DOCS_DIR}' not found")
        return
    
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.txt")) + glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    print(f"🔍 Found {len(doc_files)} documents in '{DOCS_DIR}'")
    
    if not doc_files:
        print("❌ No documents found for processing. Add files to the Documents directory.")
        return
    
    # Process documents and generate chunks
    all_chunks = process_documents(doc_files)
    
    if not all_chunks:
        print("❌ No chunks were generated. Check your documents and processing settings.")
        return
    
    # Print chunk statistics
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