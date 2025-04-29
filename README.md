# RAGbot - Proposal Generation Assistant

A lightweight Retrieval-Augmented Generation system for answering questions about harry potter using a vector database of embeddings from harry potter books.

## Requirements

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key

## Quick Start

1. **Create environment file**
   ```bash
   cp env-template.txt .env
   ```
   Edit the `.env` file with your OpenAI API key.

2. **Start the system**
   ```bash
   docker compose up -d
   ```

3. **Access the web interface**
   - Open http://localhost:3000
   - Login with the token from your .env file
   - Add a new conection to http://localhost:8000

4. **To remove all docker info for full restart**
   ```bash
   docker compose down -v
   docker system prune -a --volumes -f
   ```

## Adding Documents

1. Place `.txt` or `.pdf` files in the `Documents/` folder
2. Ingest the documents:
   ```bash
   docker compose up ragbot-ingest
   ```

## Core Features

- Vector similarity search with pgvector
- Semantic chunking of documents
- Context-aware proposal generation
- OpenWebUI integration
- Support for both text and PDF documents

## Project Structure

- `rag/` - Core components (retriever, prompt_builder, generator)
- `api.py` - FastAPI server
- `rag/ingest.py` - Document processing pipeline
- `main.py` - CLI interface

## What's Next?

- Support for additional file formats (Excel, JSON)
- Improved retrieval capabilities
- One-click deployment solution
