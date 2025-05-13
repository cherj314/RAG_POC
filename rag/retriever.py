import sys, os, json, numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import get_db_connection, release_connection, get_collection_id, EMBEDDING_MODEL

# Initialize the embedding model (lazily loaded and cached)
embed_model = None

# Get the sentence transformer model for creating embeddings
def get_embed_model():
    global embed_model
    if embed_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        embed_model = SentenceTransformer(model_name)
    return embed_model

# Cache for collection ID
collection_id_cache = None

# Get the collection ID with caching
def get_cached_collection_id(conn):
    global collection_id_cache
    if collection_id_cache is None:
        collection_id_cache = get_collection_id(conn)
    return collection_id_cache

# Calculate cosine similarity between two embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    # Convert to numpy arrays if they aren't already
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Calculate dot product
    dot_product = np.dot(embedding1, embedding2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    return float(dot_product / (magnitude1 * magnitude2))

def perform_vector_search(conn, query: str, collection_id: str, 
                         embedding: np.ndarray, k: int, 
                         similarity_threshold: float, debug: bool = False) -> List[Tuple]:
    cur = conn.cursor()
    
    # Pure vector similarity search
    sql_query = """
        SELECT document, cmetadata, 
               1 - (embedding <=> %s::vector) as similarity_score
        FROM langchain_pg_embedding
        WHERE collection_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    
    # Execute with parameters
    cur.execute(sql_query, (embedding.tolist(), collection_id, embedding.tolist(), k))
    
    # Process results
    results = []
    for row in cur.fetchall():
        doc, metadata, similarity_score = row
        if similarity_score >= similarity_threshold:
            # Process metadata
            if isinstance(metadata, str) and metadata.startswith('{'):
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
                    
            results.append((doc, metadata, similarity_score))
            
            if debug:
                source = metadata.get("file_name", "Unknown") if isinstance(metadata, dict) else "Unknown"
                print(f"Found: {source} - Score: {similarity_score:.3f}")
    
    return results

# Search function for PostgreSQL with pure vector similarity for Harry Potter
def search_postgres(query, k=5, similarity_threshold=0.3, debug=False):
    if debug:
        print(f"Searching for query: '{query}'")
        print(f"Similarity threshold: {similarity_threshold}")
    
    # Generate embedding for the query
    model = get_embed_model()
    embedding = model.encode(query)
    
    # Connect to the PostgreSQL database
    conn = get_db_connection()
    
    try:
        collection_id = get_cached_collection_id(conn)

        # Perform pure vector search
        results = perform_vector_search(
            conn=conn,
            query=query,
            collection_id=collection_id,
            embedding=embedding,
            k=k,
            similarity_threshold=similarity_threshold,
            debug=debug
        )
        
        return results
        
    except Exception as e:
        print(f"Error in search_postgres: {e}")
        import traceback
        print(traceback.format_exc())
        return []
    finally:
        # Always release the connection back to the pool
        release_connection(conn)

    if debug:
        print(f"Searching for query: '{query}'")
        print(f"Similarity threshold: {similarity_threshold}")
    
    # Generate embedding for the query
    model = get_embed_model()
    embedding = model.encode(query)
    
    # Connect to the PostgreSQL database
    conn = get_db_connection()
    
    try:
        collection_id = get_cached_collection_id(conn)

        # Perform hybrid search combining vector similarity and keyword matching
        results = perform_vector_search(
            conn=conn,
            query=query,
            collection_id=collection_id,
            embedding=embedding,
            k=k,  # Get more initial results for filtering
            similarity_threshold=similarity_threshold,
            debug=debug
        )
        
        return results
        
    except Exception as e:
        print(f"Error in search_postgres: {e}")
        import traceback
        print(traceback.format_exc())
        return []
    finally:
        # Always release the connection back to the pool
        release_connection(conn)