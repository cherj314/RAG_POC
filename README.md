# RAGbot - Proposal Generation Assistant

A lightweight Retrieval-Augmented Generation system for creating customized software proposals using a vector database of historical documents.

## Requirements

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key

## Quick Start

1. **Run the setup script**
   ```bash
   python rag/ingest.py setup
   ```
   This will:
   - Create a .env file if needed
   - Create a Documents directory
   - Check Docker installation
   - Create helper scripts for easy management

2. **Start the system**
   ```bash
   # On Linux/macOS/WSL:
   ./ragbot-start.sh
   
   # On Windows:
   ragbot-start.bat
   ```

3. **Access the web interface**
   - Open http://localhost:3000
   - Use the default token from your .env file
   - Change API connection to use http://localhost:8000 

## Adding Documents

1. Place text files (.txt) or PDF files (.pdf) in the `Documents/` folder
2. Ingest documents with:
   ```bash
   docker compose up ragbot-ingest
   ```

## Resetting the System

If you need to reset the system (clear all data and start fresh):

```bash
# On Linux/macOS/WSL:
./ragbot-reset.sh

# On Windows:
ragbot-reset.bat
```

## Core Features

- Vector similarity search with pgvector
- Semantic chunking of documents 
- Context-aware proposal generation
- OpenWebUI integration
- Support for both text (.txt) and PDF (.pdf) documents

## Project Structure

- `rag/` - Core components (retriever, prompt_builder, generator)
- `api.py` - FastAPI server for web integration
- `rag/ingest.py` - Document indexing pipeline and setup utility
- `main.py` - CLI interface

## Technology

- PostgreSQL with pgvector for vector similarity
- Sentence Transformers for embeddings
- OpenAI API for generation
- FastAPI for web server
- pypdf for PDF processing

## What's Next?

- Add other file type ingestion (xls, json)
- Increase vectordb scalability and retrieval speed
- Improve retrieval capabilities and customization
- Improve containerization - goal is 1 click deploy anywhere
- Restore CLI functionality and expose backend of vector db for testing / tuning