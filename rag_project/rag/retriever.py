from sentence_transformers import SentenceTransformer
import sys
import os

# Add the parent directory to sys.path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import get_db_connection, release_connection, get_collection_id, EMBEDDING_MODEL

# Initialize the embedding model (lazily loaded)
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
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return embed_model

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
    # Generate embedding for the query
    model = get_embed_model()
    embedding = model.encode(query)
    
    # Connect to the PostgreSQL database
    conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        collection_id = get_collection_id(conn)
        
        # Use cosine similarity search (1 - distance = similarity)
        cur.execute("""
            SELECT document, cmetadata, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM langchain_pg_embedding
            WHERE collection_id = %s
            ORDER BY similarity_score DESC
            LIMIT %s;
        """, (embedding.tolist(), collection_id, k))
        
        # Filter results by similarity threshold
        results = []
        for row in cur.fetchall():
            doc, metadata, similarity_score = row
            if similarity_score >= similarity_threshold:
                results.append((doc, metadata, similarity_score))
    finally:
        # Always release the connection back to the pool
        release_connection(conn)
        
    return results