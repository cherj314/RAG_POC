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

def search_postgres(query, k=5, similarity_threshold=0.7, debug=False):
    """
    Search for semantically similar document chunks.
    
    Args:
        query (str): The user query
        k (int): Maximum number of results to return
        similarity_threshold (float): Minimum similarity score (0-1) to include a result
        debug (bool): Whether to print debug information
        
    Returns:
        list: List of tuples containing (document_text, metadata, similarity_score)
    """
    # Generate embedding for the query
    model = get_embed_model()
    embedding = model.encode(query)
    
    # Connect to the PostgreSQL database
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        collection_id = get_cached_collection_id(conn)
        
        if debug:
            print(f"Searching for query: '{query}'")
            print(f"Using collection ID: {collection_id}")
            print(f"Similarity threshold: {similarity_threshold}")
        
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
                # If metadata is a string containing JSON, convert it to a dictionary
                if isinstance(metadata, str) and metadata.startswith('{'):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                results.append((doc, metadata, similarity_score))
                
                if debug:
                    source = metadata.get("file_name", "Unknown") if isinstance(metadata, dict) else "Unknown"
                    print(f"Found: {source} - Score: {similarity_score:.3f}")
        
        # Sort by similarity score (highest first) and limit to k results
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:k]
        
    finally:
        # Always release the connection back to the pool
        release_connection(conn)
        
    if debug:
        print(f"Found {len(results)} relevant chunks")
    
    return results