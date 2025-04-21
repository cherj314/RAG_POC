# 🧠 RAGbot — Retrieval-Augmented Generation Assistant for Proposal Generation

RAGbot is a lightweight Retrieval-Augmented Generation (RAG) assistant designed to generate software proposals using historical scopes of work stored in a PostgreSQL vector database. It helps software companies quickly create customized proposals by retrieving relevant examples from past work.

## 📦 Features

- 🔍 **Context-Aware Retrieval**: Uses `pgvector` to search relevant past documents based on semantic similarity
- 🧱 **Modular Design**: Clean separation between retrieval, prompt construction, and generation
- 🤖 **Generative AI Ready**: Works with OpenAI APIs and extendable to other LLMs
- 📊 **Flexible Similarity Thresholds**: Configure the relevance level of retrieved content
- 🌐 **Web Interface**: OpenWebUI integration for easy interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- OpenAI API key (for generation)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ragbot.git
cd ragbot

# Run setup script
python setup.py
```

The setup script will:
1. Check prerequisites
2. Create a `.env` file from `.env.example`
3. Set up a Python virtual environment
4. Install required dependencies
5. Create necessary directories

### Starting the System

```bash
# Start all services with Docker Compose
docker compose up -d

# Ingest your documents (only needed once or when documents change)
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate
python ingest.py

# Run the interactive CLI (optional)
python main.py
```

### Accessing the Web Interface

Open your browser and navigate to:
```
http://localhost:3000
```

Use the default authentication token from your `.env` file to log in.

## 📋 Usage Options

### Interactive CLI

```bash
python main.py
```

This starts the interactive command-line interface where you can directly input proposal requests.

### API Server

```bash
# Start the API server (if not using Docker)
python -m api
```

The API will be available at `http://localhost:8000`.

### Docker Deployment

All components (database, API, web UI) can be started with Docker Compose:

```bash
docker compose up -d
```

## 🗄️ Adding Your Own Documents

1. Place your text files in the `Documents/` folder
2. Run the ingestion script:
   ```bash
   python ingest.py
   ```

## 🛠️ Architecture

RAGbot follows a simple but effective workflow:

1. **Retrieval**: Searches the vector database for relevant document chunks
2. **Prompt Building**: Constructs prompts that combine retrieved context with the user query
3. **Generation**: Passes the constructed prompt to an LLM to generate a coherent proposal

## ⚙️ Configuration

Edit `.env` file to customize:

- **OpenAI API Key**: Required for generation
- **Database Settings**: Connection details
- **Embedding Model**: Default is `sentence-transformers/all-MiniLM-L6-v2`
- **Chunking Parameters**: Size and overlap of document chunks
- **Server Ports**: API and Web UI ports

## 🧩 Project Structure

```
ragbot/
├── .env.example           # Example environment variables
├── .env                   # Your configured environment variables
├── setup.py               # Setup script for easy installation
├── api.py                 # FastAPI server for OpenWebUI integration
├── main.py                # CLI entry point
├── ingest.py              # Document ingestion script
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # PostgreSQL with pgvector
├── Dockerfile-api         # API server container
├── db-setup.sql           # Database initialization SQL
├── requirements.txt       # Python dependencies
├── venv/                  # Python virtual environment
├── Documents/             # Your document storage
│   └── project_proposals.txt  # Sample document
└── rag/                   # Core RAG components
    ├── config.py          # Configuration and database connection
    ├── retriever.py       # Vector retrieval functionality
    ├── prompt_builder.py  # Prompt engineering
    └── generator.py       # LLM integration
```

## 🧪 Customization Options

### Using a Different LLM

To switch from OpenAI to another provider:

1. Modify the `generator.py` file to use your preferred API
2. Update the environment variables accordingly

### Embedding Model

You can change the embedding model in your `.env` file:

```
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Adding More Document Types

The system currently processes text files. To add support for PDFs, DOCs, etc.:

1. Install additional dependencies for document processing
2. Modify the `ingest.py` file to handle these file types

## 🔒 Security Notes

- The `.env` file contains sensitive information including API keys
- Ensure it's added to `.gitignore` and not shared publicly
- For production deployment, use proper secrets management

## 📚 Troubleshooting

### Common Issues

1. **Database Connection Errors**: 
   - Ensure PostgreSQL is running
   - Check your DB credentials in `.env`

2. **OpenAI API Errors**:
   - Verify your API key is correct
   - Check for sufficient API credits

3. **Ingestion Problems**:
   - Make sure the `Documents` directory exists
   - Check file permissions

For more detailed troubleshooting, check the API logs:
```bash
docker compose logs ragbot-api
```

## 🙏 Acknowledgments

- Built with [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Uses [SentenceTransformers](https://www.sbert.net/) for embeddings
- Integration with [OpenWebUI](https://github.com/open-webui/open-webui)