import sys
import os
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import get_db_connection, release_connection, get_collection_id, EMBEDDING_MODEL

# Initialize the embedding model (lazily loaded and cached)
embed_model = None

def get_embed_model():
    """Get the sentence transformer model for creating embeddings."""
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return embed_model

# Cache for collection ID
collection_id_cache = None

def get_cached_collection_id(conn):
    """Get the collection ID with caching."""
    global collection_id_cache
    if collection_id_cache is None:
        collection_id_cache = get_collection_id(conn)
    return collection_id_cache

def search_postgres(query, k=5, similarity_threshold=0.7):
    """Search for semantically similar document chunks."""
    # Generate embedding for the query
    model = get_embed_model()
    embedding = model.encode(query)
    
    # Connect to the PostgreSQL database
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        collection_id = get_cached_collection_id(conn)
        
        # Use cosine similarity search with index optimization
        cur.execute("""
            SELECT document, cmetadata, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM langchain_pg_embedding
            WHERE collection_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding.tolist(), collection_id, embedding.tolist(), k * 2))
        
        # Filter results by similarity threshold
        results = []
        for row in cur.fetchall():
            doc, metadata, similarity_score = row
            if similarity_score >= similarity_threshold:
                results.append((doc, metadata, similarity_score))
        
        # Sort by similarity score (highest first) and limit to k results
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:k]
        
    finally:
        # Always release the connection back to the pool
        release_connection(conn)
        
    return results