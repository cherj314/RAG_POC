-- Performance optimization for the pgvector setup

-- Enable the vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Make sure the embedding column is the right type
ALTER TABLE langchain_pg_embedding
ALTER COLUMN embedding TYPE vector(384);

-- Check if index exists before creating
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE indexname = 'langchain_pg_embedding_vector_idx'
    ) THEN
        -- Create an IVFFlat index for faster similarity searches
        -- This index is optimized for approximate nearest neighbor (ANN) searches
        -- Lists=100 is a good starting point for a few thousand vectors
        CREATE INDEX langchain_pg_embedding_vector_idx
        ON langchain_pg_embedding
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        
        RAISE NOTICE 'IVFFlat vector index created successfully';
    ELSE
        RAISE NOTICE 'Vector index already exists';
    END IF;
END
$$;

-- Add an index on collection_id to speed up filtering
-- This helps when we have multiple collections in the same table
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE indexname = 'langchain_pg_embedding_collection_id_idx'
    ) THEN
        CREATE INDEX langchain_pg_embedding_collection_id_idx
        ON langchain_pg_embedding(collection_id);
        
        RAISE NOTICE 'Collection ID index created successfully';
    ELSE
        RAISE NOTICE 'Collection ID index already exists';
    END IF;
END
$$;

-- Analyze the tables to update statistics for the query planner
ANALYZE langchain_pg_embedding;

-- Increase work_mem for better sort/join performance during vector queries
-- This is a session setting that should be applied in production settings
ALTER SYSTEM SET work_mem = '32MB';

-- If you have enough RAM, increase shared_buffers for better caching
-- ALTER SYSTEM SET shared_buffers = '1GB';

-- Reload configuration to apply system changes
SELECT pg_reload_conf();

-- Display current settings
SELECT name, setting, unit, context 
FROM pg_settings 
WHERE name IN ('work_mem', 'shared_buffers');

-- Check the created indexes
SELECT
    indexname,
    indexdef
FROM
    pg_indexes
WHERE
    tablename = 'langchain_pg_embedding';
