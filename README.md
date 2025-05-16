<a id="readme-top"></a>

# RAG_POC: Harry Potter Retrieval Augmented Generation System

## Create a Harry Potter Expert that uses Citations

[![GitHub issues](https://img.shields.io/github/issues/cherj314/RAG_POC.svg)](https://github.com/cherj314/RAG_POC/issues)

## About The Project

RAG_POC is a Retrieval-Augmented Generation (RAG) system designed to provide accurate answers to questions about the Harry Potter series. The system ingests Harry Potter books, processes them into semantically meaningful chunks, stores them in a vector database, and uses modern language models to generate precise answers based on the retrieved context.

This proof-of-concept demonstrates how RAG can be used to create a knowledge base that provides accurate, source-backed answers while avoiding hallucinations common in large language models.

### Built With

* [![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
* [![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0+-green.svg)](https://fastapi.tiangolo.com/)
* [![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
* [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](https://github.com/pgvector/pgvector)
* [![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-2.2.0+-orange.svg)](https://www.sbert.net/)
* [![LangChain](https://img.shields.io/badge/LangChain-0.0.267+-yellow.svg)](https://langchain.readthedocs.io/)
* [![OpenAI API](https://img.shields.io/badge/OpenAI-API-teal.svg)](https://openai.com/)
* [![Ollama](https://img.shields.io/badge/Ollama-LLM--integration-purple.svg)](https://ollama.ai/)
* [![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Interface-blue.svg)](https://github.com/open-webui/open-webui)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Follow these steps to set up a local copy of the project.

### Prerequisites

* wsl2 (Ubuntu)
* Docker and Docker Compose
* OpenAI API key (if using OpenAI models)
* Git

### Installation

1. Clone the repository in Ubuntu environment with prerequisites installed
   ```sh
   git clone https://github.com/cherj314/RAG_POC.git
   cd RAG_POC
   ```

2. Create a `.env` file based on the template and add your OpenAI API key
   ```sh
   cp env-template.txt .env
   OPENAI_API_KEY=your_api_key_here
   ```

3. Add your Harry Potter or other books/files in text (.txt) or PDF (.pdf) format to the `Documents` directory
   ```sh
   mkdir -p Documents
   # Copy your Harry Potter books into the Documents directory
   ```

4. Start the services with Docker Compose
   ```sh
   docker compose up
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

The first build can take 10-15 minutes depending on hardware and internet connection, subsequent builds are fast.

Once the system is up and running:

1. Access the web interface at `http://localhost:3000` (or the port specified in your `.env` file)
2. Add a connection to http://localhost:8000 to access ollama and your openai API key to access openAI models
3. Start asking questions about Harry Potter!


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

The system consists of the following components:

1. **Document Processing:** PDF and text files are loaded, preprocessed, and split into semantically meaningful chunks.

2. **Vector Database:** PostgreSQL with pgvector extension stores document chunks as embeddings for semantic search.

3. **API Server:** FastAPI provides endpoints for querying the system and retrieving information.

4. **LLM Integration:** Supports both OpenAI models and local models via Ollama.

5. **Web Interface:** OpenWebUI provides a chat interface for interacting with the system.

Key modules:

- `ingest.py`: Processes documents and stores them in the vector database
- `api.py`: Provides API endpoints for querying the system
- `retriever.py`: Handles vector similarity search in the database
- `generator.py`: Generates responses using LLMs
- `semantic_text_splitter.py`: Splits text into semantically coherent chunks

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

- **Semantic Chunking:** Documents are split into chunks based on semantic similarity, not just token counts
- **Source Attribution:** Responses cite the specific passages they're based on
- **Multi-Modal Support:** Works with both PDF and text files
- **Multiple LLM Options:** Use OpenAI models or local models via Ollama
- **Efficient Retrieval:** PostgreSQL with pgvector provides fast semantic search
- **Dockerized Deployment:** Easy setup with Docker Compose
- **User-Friendly Interface:** Simple chat interface via OpenWebUI

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration

The system can be configured through the `.env` file. Key configuration options include:

### Embedding Model - Use alternative for better performance at cost of speed
```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Alternative: EMBEDDING_MODEL=all-mpnet-base-v2
```

### Chunking Configuration - Adjust and test to improve performance
```
MIN_CHUNK_SIZE=200
MAX_CHUNK_SIZE=2000
SEMANTIC_SIMILARITY=0.6
RESPECT_STRUCTURE=true
CHUNK_OVERLAP=100
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>