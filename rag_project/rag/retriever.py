import psycopg2
from sentence_transformers import SentenceTransformer
import sys
import os
import numpy as np
from sklearn.preprocessing import normalize

# Add the parent directory to sys.path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import *

# Initialize the embedding model
embed_model = None

def get_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return embed_model

def search_postgres(query, k=5, similarity_threshold=0.7):
    # Generate the embedding for the query
    model = get_embed_model() # Get the model only when needed
    embedding = model.encode(query)
    
    # Connect to the PostgreSQL database
    conn = get_db_connection()

    cur = conn.cursor()

    # First get the collection_id for your collection name
    cur.execute("""
        SELECT uuid FROM langchain_pg_collection 
        WHERE name = %s;
    """, (COLLECTION_NAME,))

    collection_id = get_collection_id(conn)

    # Use proper cosine similarity - the <=> operator returns distance, not similarity
    # Lower distance values mean higher similarity
    cur.execute("""
        SELECT document, cmetadata, 
               1 - (embedding <=> %s::vector) as similarity_score
        FROM langchain_pg_embedding
        WHERE collection_id = %s
        ORDER BY similarity_score DESC
        LIMIT %s;
    """, (embedding.tolist(), collection_id, k))

    # Fetch results and filter by similarity threshold
    results = []
    for row in cur.fetchall():
        doc, metadata, similarity_score = row
        if similarity_score >= similarity_threshold:
            results.append((doc, metadata, similarity_score))
    
    # Close the database connection
    release_connection(conn)

    return results