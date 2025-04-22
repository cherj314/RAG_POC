import atexit
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize():
    """Initialize system components"""
    print("🚀 Initializing RAGbot...")
    
    # Import and initialize database connection pool
    from rag.config import init_db_pool
    init_db_pool()
    
    # Preload the embedding model
    from rag.retriever import get_embed_model
    get_embed_model()
    
    print("✅ RAGbot ready!")

def cleanup():
    """Clean up resources on exit"""
    from rag.config import DB_POOL
    if DB_POOL is not None:
        DB_POOL.closeall()
        print("🧹 Cleaned up database connections")

def generate_proposal(user_input):
    """Generate a proposal based on user input using RAG methodology."""
    from rag.retriever import search_postgres
    from rag.prompt_builder import build_prompt
    from rag.generator import generate_response
    
    # Step 1: Retrieve relevant chunks using vector similarity
    print("🔍 Searching for relevant content...")
    chunks = search_postgres(user_input, k=5, similarity_threshold=0.5)
    
    # Step 2: Handle case where no relevant content found
    if not chunks:
        print("❌ No relevant content found. Please try again with a different query.")
        return None
    
    # Step 3: Build prompt with retrieved context and user input
    prompt = build_prompt(chunks, user_input)
    
    # Step 4: Display retrieved information
    print("\n🔍 Retrieved Chunks:\n")
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        print(f"[{i}] Score: {score:.4f}")
        print(f"{doc[:150]}...\n")
    
    # Step 5: Generate response
    print("\n🧠 Generating proposal...\n")
    response = generate_response(prompt)
    print(response)
    
    return response

def run_interactive():
    """Run the RAGbot in interactive mode"""
    print("\n🤖 Welcome to RAGbot - Your AI Proposal Assistant!")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        user_query = input("Enter a proposal request:\n> ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        
        print("\nGenerating proposal...\n")
        generate_proposal(user_query)
        print("\n---\n")

if __name__ == "__main__":
    # Register cleanup function to run on exit
    atexit.register(cleanup)
    
    # Initialize components
    initialize()
    
    if len(sys.argv) > 1:
        # Non-interactive mode with command line argument
        generate_proposal(sys.argv[1])
    else:
        # Interactive mode
        run_interactive()