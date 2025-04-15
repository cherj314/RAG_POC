🧠 RAGbot — Lightweight RAG Assistant for Proposal Generation
RAGbot is a lightweight Retrieval-Augmented Generation (RAG) assistant designed to generate software proposals using historical scopes of work stored in a PostgreSQL vector database.

📦 Features
🔍 Context-Aware Retrieval: Uses pgvector to search relevant past documents based on semantic similarity.
🧱 Modular Design: Clean separation between retrieval, prompt construction, and generation.
🤖 Generative AI Ready: Easily extendable to plug into a local LLM like LLaMA or any API-based model.

⚙️ Setup
1. Install Dependencies
Make sure pgvector is installed on your PostgreSQL server:
CREATE EXTENSION IF NOT EXISTS vector;

2. Configure Environment
Edit rag/config.py to set:
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASSWORD = "your_pass"
DB_HOST = "localhost"
DB_PORT = "5432"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # or your preferred SentenceTransformer

3. Run the Assistant
python main.py
You’ll be prompted to enter a proposal topic. The app will fetch relevant chunks, build a prompt, and (optionally) pass it to your LLM for response generation.

🔮 Coming Soon
🔌 LLaMA/llama.cpp integration
📤 Response generation using the retrieved context
🧪 CLI interface for interactive use using openwebUI
