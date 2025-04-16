from sentence_transformers import SentenceTransformer
import sys
import os
import time

# Add the parent directory to sys.path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import get_db_connection, release_connection, get_collection_id, EMBEDDING_MODEL

# Initialize the embedding model (lazily loaded and cached)
embed_model = None

def get_embed_model():
    """
    Get the sentence transformer model for creating embeddings.
    Initializes the model if it doesn't exist yet.
    
    Returns:
        SentenceTransformer: The embedding model
    """
    global embed_model
    if embed_model is None:
        print("🔄 Loading embedding model...")
        start_time = time.time()
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
    return embed_model

# Cache for collection ID to avoid repeated database lookups
collection_id_cache = None

def get_cached_collection_id(conn):
    """
    Get the collection ID with caching to avoid repeated database lookups.
    
    Args:
        conn: Database connection
        
    Returns:
        str: Collection ID
    """
    global collection_id_cache
    if collection_id_cache is None:
        collection_id_cache = get_collection_id(conn)
    return collection_id_cache

def search_postgres(query, k=5, similarity_threshold=0.7):
    """
    Search the PostgreSQL database for semantically similar document chunks.
    
    Args:
        query (str): The search query
        k (int): Maximum number of results to return
        similarity_threshold (float): Minimum similarity score (0-1)
        
    Returns:
        list: List of tuples containing (document_text, metadata, similarity_score)
    """
    # Start timing for performance analysis
    start_time = time.time()
    
    # Generate embedding for the query
    model = get_embed_model()
    embedding_start = time.time()
    embedding = model.encode(query)
    embedding_time = time.time() - embedding_start
    
    # Connect to the PostgreSQL database
    db_connect_start = time.time()
    conn = get_db_connection()
    db_connect_time = time.time() - db_connect_start
    
    try:
        cursor_start = time.time()
        cur = conn.cursor()
        collection_id = get_cached_collection_id(conn)
        
        # Use cosine similarity search with index optimization
        # Use LIMIT to reduce the number of rows processed
        search_start = time.time()
        cur.execute("""
            SELECT document, cmetadata, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM langchain_pg_embedding
            WHERE collection_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding.tolist(), collection_id, embedding.tolist(), k * 2))  # Fetch more results to ensure we have enough after filtering
        search_time = time.time() - search_start
        
        # Filter results by similarity threshold
        results = []
        for row in cur.fetchall():
            doc, metadata, similarity_score = row
            if similarity_score >= similarity_threshold:
                results.append((doc, metadata, similarity_score))
        
        # Sort by similarity score (highest first) and limit to k results
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:k]
        
        total_time = time.time() - start_time
        print(f"⏱️ Search completed in {total_time:.2f}s (Embedding: {embedding_time:.2f}s, DB: {db_connect_time:.2f}s, Search: {search_time:.2f}s)")
        
    finally:
        # Always release the connection back to the pool
        release_connection(conn)
        
    return results