# RAGbot - Harry Potter Knowledge Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system designed specifically for answering questions about the Harry Potter series. RAGbot uses semantic search and vector embeddings to retrieve relevant passages from the Harry Potter books and generate accurate, contextual responses.

## Features

- **Semantic Search**: Uses sophisticated vector embeddings to find the most relevant text passages
- **Smart Document Chunking**: Intelligently chunks documents based on narrative structure and semantic similarity
- **Hybrid Retrieval**: Combines vector similarity with keyword matching for improved results
- **Multiple LLM Support**: Compatible with both OpenAI models and local Ollama models
- **Dialogue Preservation**: Special handling to keep dialogue exchanges together during chunking
- **PDF Support**: Optimized handling for PDF files with structure recognition
- **Web Interface**: Integration with OpenWebUI for easy interaction
- **Docker Deployment**: Full containerization for easy setup and deployment
- **Robust Error Handling**: Multiple fallback mechanisms for reliable document processing

## Requirements

- Docker and Docker Compose
- OpenAI API key (optional if using only Ollama models)
- Python 3.9+ (for development without Docker)
- WSL (Windows Subsystem for Linux) if running on Windows

## Quick Start

1. **Create environment file**
   ```bash
   cp env.template.txt .env
   ```
   Edit the `.env` file with your OpenAI API key and other configurations.
   
   > **Note for Windows users**: This project works best with WSL (Windows Subsystem for Linux). Use Ubuntu under WSL for the best experience.

2. **Start the system**
   ```bash
   docker compose up -d
   ```
   This will start the PostgreSQL database with pgvector extension, the RAGbot API server, the Ollama server (for local models), and the OpenWebUI interface.

3. **Ingest documents**
   ```bash
   docker compose up ragbot-ingest
   ```
   This will process and index any Harry Potter text or PDF files in the `Documents/` folder.

4. **Access the web interface**
   - Open http://localhost:3000 in your browser
   - Login with the token from your `.env` file (WEBUI_AUTH_TOKEN)
   - Add a new connection to http://localhost:8000
   - Start asking questions about Harry Potter!

5. **To remove all Docker data for a full restart**
   ```bash
   docker compose down -v
   docker system prune -a --volumes -f
   ```

## Adding Documents

1. Place `.txt` or `.pdf` files containing Harry Potter book content in the `Documents/` folder
2. Run the ingestion process:
   ```bash
   docker compose up ragbot-ingest
   ```

## System Architecture

### Core Components

- **Vector Database**: PostgreSQL with pgvector extension for efficient similarity search
- **Document Processing**: Enhanced semantic text splitter optimized for narrative text
- **Embedding Model**: Sentence transformers for high-quality text embeddings
- **Retrieval Engine**: Hybrid search combining vector similarity and keyword matching
- **Generation Layer**: Integration with OpenAI and Ollama models for response generation
- **API Server**: FastAPI backend with chat completion endpoints compatible with OpenAI format

### Project Structure

- `rag/config.py` - Database and configuration management
- `rag/ingest.py` - Document processing pipeline
- `rag/generator.py` - LLM integration for generating responses
- `rag/retriever.py` - Vector and keyword search implementation
- `rag/prompt_builder.py` - Context formatting for the LLM
- `rag/semantic_text_splitter.py` - Narrative-aware document chunking
- `rag/pdf_loader.py` - PDF handling with structure recognition
- `api.py` - FastAPI server with OpenAI-compatible endpoints
- `docker-compose.yml` - Container orchestration

## API Endpoints

- **POST /api/chat/completions** - OpenAI-compatible chat completion endpoint
- **GET /api/models** - List available models
- **POST /api/generate-proposal** - Generate a response for a specific query
- **POST /api/retrieve-chunks** - Get only the retrieved chunks without generation
- **GET /api/health** - Health check endpoint

## Configuration Options

The system can be configured through the `.env` file:

- **Embedding Model**: Choose between different sentence transformer models
- **Chunking Strategy**: Use semantic or fixed-size chunking
- **LLM Selection**: Configure OpenAI or Ollama models
- **Database Settings**: Customize PostgreSQL connection settings
- **Processing Parameters**: Adjust batch size, workers, chunk size and overlap

## Future Improvements

- Support for additional file formats (DOCX, EPUB)
- Improved document segmentation for complex narratives
- User feedback integration for retrieval quality