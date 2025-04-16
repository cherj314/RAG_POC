from dotenv import load_dotenv
import time
import atexit
import sys

# Load environment variables
load_dotenv()

def initialize():
    """Initialize system components"""
    print("🚀 Initializing RAGbot...")
    
    # Import and initialize database connection pool
    from rag.config import init_db_pool
    init_db_pool()
    
    # Preload the embedding model to avoid delay during first query
    from rag.retriever import get_embed_model
    get_embed_model()
    
    print("✅ RAGbot ready!")

def cleanup():
    """Clean up resources on exit"""
    # Import here to avoid circular imports
    from rag.config import DB_POOL
    if DB_POOL is not None:
        DB_POOL.closeall()
        print("🧹 Cleaned up database connections")

def generate_proposal(user_input):
    """
    Generate a proposal based on user input using RAG methodology.
    
    Args:
        user_input (str): The user's request for a proposal
        
    Returns:
        str or None: Generated proposal text if successful, None otherwise
    """
    from rag.retriever import search_postgres
    from rag.prompt_builder import build_prompt
    from rag.generator import generate_response  # Import the generator function
    
    # Record start time for performance monitoring
    start_time = time.time()
    
    # Step 1: Retrieve relevant chunks using vector similarity
    k = 5  # Number of chunks to retrieve
    similarity_threshold = 0.5  # Minimum similarity score
    
    print("🔍 Searching for relevant content...")
    search_start = time.time()
    chunks = search_postgres(user_input, k=k, similarity_threshold=similarity_threshold)
    search_time = time.time() - search_start
    
    # Step 2: Handle case where no relevant content found
    if not chunks:
        print("❌ No relevant content found. Please try again with a different query.")
        return None
    
    # Step 3: Build prompt with retrieved context and user input
    prompt_start = time.time()
    prompt = build_prompt(chunks, user_input)
    prompt_time = time.time() - prompt_start
    
    # Step 4: Display retrieved information for transparency
    print("\n🔍 Retrieved Chunks:\n")
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        print(f"[{i}] Score: {score:.4f}")
        print(f"{doc[:150]}...\n")
    
    print("\n🧠 Prompt Preview:\n")
    print(f"{prompt[:500]}...\n")
    
    # Step 5: Generate response using GPT-4o
    print("\n🧠 Generating proposal with GPT-4o...\n")
    generation_start = time.time()
    response = generate_response(prompt)
    generation_time = time.time() - generation_start
    print(f"\n✅ Response generated in {generation_time:.2f}s\n")
    print(response)
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total process completed in {total_time:.2f}s")
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