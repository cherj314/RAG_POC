# RAGbot - Proposal Generation Assistant

A lightweight Retrieval-Augmented Generation system for creating customized software proposals using a vector database of historical documents.

## Requirements

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key

## Quick Start

1. **Setup environment**
   ```bash
   python setup.py
   ```

2. **Start the system**
   ```bash
   docker compose up -d
   ```

3. **Access the web interface**
   - Open http://localhost:3000
   - Use the default token from your .env file
   - Change api connection to use http://localhost:8000 

## Core Features

- Vector similarity search with pgvector
- Semantic chunking of documents 
- Context-aware proposal generation
- OpenWebUI integration

## Project Structure

- `rag/` - Core components (retriever, prompt_builder, generator)
- `api.py` - FastAPI server for web integration
- `ingest.py` - Document indexing pipeline
- `main.py` - CLI interface

## Adding Documents

1. Place text files in the `Documents/` folder
2. Run ingestion:
   ```bash
   docker compose up ragbot-ingest
   ```

## Technology

- PostgreSQL with pgvector for vector similarity
- Sentence Transformers for embeddings
- OpenAI API for generation
- FastAPI for web server