-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Make sure the embedding column is the right type (only if table exists)
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'langchain_pg_embedding'
    ) THEN
        ALTER TABLE langchain_pg_embedding
        ALTER COLUMN embedding TYPE vector(384);
    END IF;
END
$$;

-- Check if index exists before creating
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'langchain_pg_embedding'
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE indexname = 'langchain_pg_embedding_vector_idx'
    ) THEN
        -- Create an IVFFlat index for faster similarity searches
        CREATE INDEX langchain_pg_embedding_vector_idx
        ON langchain_pg_embedding
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    END IF;
END
$$;

-- Add an index on collection_id to speed up filtering
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'langchain_pg_embedding'
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE indexname = 'langchain_pg_embedding_collection_id_idx'
    ) THEN
        CREATE INDEX langchain_pg_embedding_collection_id_idx
        ON langchain_pg_embedding(collection_id);
    END IF;
END
$$;

-- Analyze the tables to update statistics for the query planner (only if they exist)
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'langchain_pg_embedding'
    ) THEN
        ANALYZE langchain_pg_embedding;
    END IF;
END
$$;

-- Increase work_mem for better performance
ALTER SYSTEM SET work_mem = '32MB';

-- Reload configuration to apply system changes
SELECT pg_reload_conf();