import sys, os, json
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import get_db_connection, release_connection, get_collection_id, EMBEDDING_MODEL

# Use module-level variables for caching
embed_model = None
collection_id_cache = None

# Lazily load the embedding model
def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL))
    return embed_model

# Search for relevant document chunks using vector similarity
def search_postgres(query, k=5, similarity_threshold=0.3, debug=False):
    if debug:
        print(f"Searching for: '{query}' (threshold: {similarity_threshold})")
    
    # Generate query embedding
    embedding = get_embed_model().encode(query)
    
    # DB connection
    conn = get_db_connection()
    try:
        # Get collection ID (with caching)
        global collection_id_cache
        if collection_id_cache is None:
            collection_id_cache = get_collection_id(conn)
        
        # Execute vector search
        cur = conn.cursor()
        cur.execute(
            """
            SELECT document, cmetadata, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM langchain_pg_embedding
            WHERE collection_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """, 
            (embedding.tolist(), collection_id_cache, embedding.tolist(), k)
        )
        
        # Process results
        results = []
        for doc, metadata, score in cur.fetchall():
            if score >= similarity_threshold:
                # Parse metadata if needed
                if isinstance(metadata, str) and metadata.startswith('{'):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                        
                results.append((doc, metadata, score))
                
                if debug:
                    source = metadata.get("file_name", "Unknown") if isinstance(metadata, dict) else "Unknown"
                    print(f"Found: {source} - Score: {score:.3f}")
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        print(traceback.format_exc())
        return []
    finally:
        release_connection(conn)