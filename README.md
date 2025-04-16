# 🧠 RAGbot — Lightweight RAG Assistant for Proposal Generation

RAGbot is a lightweight Retrieval-Augmented Generation (RAG) assistant designed to generate software proposals using historical scopes of work stored in a PostgreSQL vector database. It helps software companies quickly create customized proposals by retrieving relevant examples from past work.

## 📦 Features

- 🔍 **Context-Aware Retrieval**: Uses `pgvector` to search relevant past documents based on semantic similarity
- 🧱 **Modular Design**: Clean separation between retrieval, prompt construction, and generation
- 🤖 **Generative AI Ready**: Easily extendable to plug into a local LLM like LLama 3.2 or any API-based model
- 📊 **Flexible Similarity Thresholds**: Configure the relevance level of retrieved content

## 🛠️ Architecture

RAGbot follows a simple but effective workflow:

1. **Retrieval**: Searches the vector database for relevant document chunks
2. **Prompt Building**: Constructs prompts that combine retrieved context with the user query
3. **Generation**: Passes the constructed prompt to an LLM to generate a coherent proposal

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ragbot.git
   cd ragbot
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start the PostgreSQL database with pgvector**
   ```bash
   docker-compose up -d
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Database Setup

1. **Connect to the database**
   ```bash
   docker exec -it $(docker ps -q -f name=postgres) psql -U myuser -d vectordb
   ```

2. **Create the pgvector extension**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Optimize vector searches with an index**
   ```sql
   ALTER TABLE langchain_pg_embedding
   ALTER COLUMN embedding TYPE vector(384);
   
   CREATE INDEX ON langchain_pg_embedding
   USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

## 💾 Data Ingestion

To populate the vector database with your proposal documents:

1. **Place your documents in the `Documents` folder**
   - Text files (.txt) with past proposals or project capabilities

2. **Run the ingestion script**
   ```bash
   python ingest.py
   ```

## 🔎 Using RAGbot

1. **Run the main application**
   ```bash
   python main.py
   ```

2. **Enter your proposal request**
   ```
   Enter a proposal request:
   > Web application for inventory management with barcode scanning
   ```

3. **Review the retrieved content and generated proposal**
   - RAGbot will display the most relevant chunks from past proposals
   - The final prompt constructed for the LLM will be shown
   - If an LLM is connected, the generated proposal will be displayed

## ⚙️ Configuration

Edit `rag/config.py` or your `.env` file to customize:

- **Database Connection**: Host, port, credentials
- **Embedding Model**: Default is `sentence-transformers/all-MiniLM-L6-v2`
- **Chunking Parameters**: Size and overlap of document chunks
- **LLama Model Path**: Path to your local LLama model

## 🔮 Connecting an LLM

### Local LLama Setup

Uncomment the LLM generation code in `main.py` after configuring:

1. **Update your `.env` file with LLama paths**
   ```
   LLAMA_CLI_PATH=/path/to/llama-cli.exe
   LLAMA_MODEL_PATH=/path/to/Llama-3.2-3B-Instruct-Q5_K_L.gguf
   ```

2. **Test generation**
   ```bash
   python main.py
   ```

### OpenAI API Connection

To use OpenAI models:

1. **Set your API key in `.env`**
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. **Modify `generator.py` to use OpenAI**

## 🧩 Project Structure

- `main.py` - Main entry point
- `rag/retriever.py` - Handles vector similarity search
- `rag/prompt_builder.py` - Constructs prompts from retrieved content
- `rag/generator.py` - Interfaces with language models
- `rag/config.py` - Configuration settings
- `ingest.py` - Data ingestion pipeline

## 🔒 Security Notes

- The `.env` file contains sensitive information including API keys
- Ensure it's added to `.gitignore` and not shared publicly
- Use environment variables in production environments

## 📈 Roadmap

- [ ] Web UI for interactive proposal generation
- [ ] Multi-model support (GPT-4, Claude, etc.)
- [ ] Proposal templates and formatting options
- [ ] Fine-tuning capabilities for domain-specific proposals

## 🙏 Acknowledgments

- Built with [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- Uses [SentenceTransformers](https://www.sbert.net/) for embeddings
- Compatible with [LLama.cpp](https://github.com/ggerganov/llama.cpp) for local inference
