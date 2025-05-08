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

# Extract key terms from the query for keyword matching
def extract_key_terms(query: str) -> List[str]:
    # Remove stop words and punctuation
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 
                 'as', 'into', 'like', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 'of', 
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'that', 'these',
                 'those', 'this', 'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how'}
    
    # Extract Harry Potter specific character names and important terms
    hp_entities = {'harry', 'potter', 'ron', 'hermione', 'dumbledore', 'snape', 'hagrid', 'voldemort',
                  'hogwarts', 'gryffindor', 'slytherin', 'ravenclaw', 'hufflepuff', 'wand', 'spell',
                  'quidditch', 'horcrux', 'deatheater', 'ministry', 'magic'}
    
    # Clean and tokenize the query
    cleaned_query = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in query.lower())
    words = cleaned_query.split()
    
    # Extract words that are not stopwords or are HP entities
    important_words = [word for word in words if word not in stop_words or word in hp_entities]
    
    # Add bigrams for important phrases
    bigrams = []
    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i+1]
        if any(entity in bigram for entity in hp_entities):
            bigrams.append(bigram)
    
    return important_words + bigrams

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

# Perform a hybrid search combining vector similarity and keyword matching
def perform_hybrid_search(conn, query: str, collection_id: str, 
                         embedding: np.ndarray, k: int, 
                         similarity_threshold: float, debug: bool = False) -> List[Tuple]:
    cur = conn.cursor()
    
    # Extract key terms for keyword search
    keywords = extract_key_terms(query)
    
    # If we have keywords, we'll use them to boost results
    if keywords and len(keywords) > 0:
        # Create keyword conditions with proper SQL escaping
        keyword_parts = []
        for keyword in keywords[:5]:  # Limit to top 5 keywords for performance
            # Escape single quotes for SQL
            safe_keyword = keyword.replace("'", "''")
            keyword_parts.append(f"document ILIKE '%{safe_keyword}%'")
        
        if keyword_parts:
            # Standard vector similarity search with keyword boost
            keyword_clause = " OR ".join(keyword_parts)
            sql_query = f"""
                SELECT document, cmetadata, 
                       1 - (embedding <=> %s::vector) as vector_score
                FROM langchain_pg_embedding
                WHERE collection_id = %s
                ORDER BY vector_score DESC
                LIMIT %s;
            """
            
            # Execute with parameters (simplified parameter list)
            cur.execute(sql_query, (embedding.tolist(), collection_id, k))
            
            # Process results with combined scoring
            results = []
            for row in cur.fetchall():
                doc, metadata, vector_score = row
                
                # Add keyword boost if document contains any keywords
                keyword_boost = 0.0
                doc_lower = doc.lower()
                for keyword in keywords:
                    if keyword.lower() in doc_lower:
                        keyword_boost = 0.2
                        break
                
                combined_score = vector_score + keyword_boost
                
                if combined_score >= similarity_threshold:
                    # Process metadata
                    if isinstance(metadata, str) and metadata.startswith('{'):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                            
                    results.append((doc, metadata, combined_score))
                    
                    if debug:
                        source = metadata.get("file_name", "Unknown") if isinstance(metadata, dict) else "Unknown"
                        print(f"Found (hybrid): {source} - Vector: {vector_score:.3f}, Boost: {keyword_boost}, Combined: {combined_score:.3f}")
    else:
        # Fallback to standard vector search if no keywords
        cur.execute("""
            SELECT document, cmetadata, 
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM langchain_pg_embedding
            WHERE collection_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding.tolist(), collection_id, embedding.tolist(), k * 2))
        
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
                    print(f"Found (vector): {source} - Score: {similarity_score:.3f}")
    
    return results

# Search function for PostgreSQL with enhanced retrieval for Harry Potter
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

        # Perform hybrid search combining vector similarity and keyword matching
        results = perform_hybrid_search(
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