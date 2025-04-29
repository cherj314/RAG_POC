import atexit
import sys
import time
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

def generate_proposal(user_input, show_retrieval_only=False):
    """Generate a proposal based on user input using RAG methodology.
    
    Args:
        user_input (str): The user's query or request
        show_retrieval_only (bool): If True, only show retrieval results without generating a proposal
    
    Returns:
        str or None: The generated proposal or None if no relevant content found
    """
    from rag.retriever import search_postgres
    from rag.prompt_builder import build_prompt
    from rag.generator import generate_response
    
    # Step 1: Retrieve relevant chunks using vector similarity
    print("🔍 Searching for relevant content...")
    chunks = search_postgres(user_input, k=5, similarity_threshold=0.3)
    
    # Step 2: Handle case where no relevant content found
    if not chunks:
        print("❌ No relevant content found. Please try again with a different query.")
        return None
    
    # Step 3: Display retrieved information
    print("\n📚 Retrieved Information:\n")
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        source = metadata.get("file_name", "Unknown") if metadata else "Unknown"
        print(f"[{i}] Source: {source} (Score: {score:.3f})")
        # Display a preview of the content (first 150 characters)
        print(f"{doc[:300]}...\n" if len(doc) > 300 else f"{doc}\n")
    
    # If user only wants to see retrieved chunks, return here
    if show_retrieval_only:
        return None
    
    # Step 4: Build prompt with retrieved context and user input
    print("\n🧠 Building prompt with retrieved context...")
    prompt = build_prompt(chunks, user_input)
    
    # Step 5: Generate response
    print("\n🧠 Generating proposal...\n")
    print("(This may take a moment depending on the complexity of your request)")
    
    start_time = time.time()
    response = generate_response(prompt, preserve_formatting=True)
    generation_time = time.time() - start_time
    
    print(f"\n✅ Proposal generated in {generation_time:.2f} seconds")
    print("\n====== GENERATED PROPOSAL ======\n")
    
    # Print the response with proper formatting
    # Split by paragraphs and print each paragraph separately
    paragraphs = response.split('\n\n')
    for paragraph in paragraphs:
        # Handle headers and bullet points with special formatting
        if paragraph.strip().startswith('#'):
            # This is a header
            print(f"\033[1m{paragraph}\033[0m")  # Bold headers if terminal supports it
        elif any(line.strip().startswith(('-', '*', '+')) for line in paragraph.split('\n')):
            # This contains bullet points
            print(paragraph)
        else:
            # Regular paragraph
            print(paragraph)
        print()  # Add empty line between paragraphs
    
    print("================================\n")
    
    return response

def run_interactive():
    """Run the RAGbot in interactive mode"""
    print("\n🤖 Welcome to RAGbot - Your AI Proposal Assistant!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'retrieve only' before your query to see only retrieved information without generating a proposal.\n")
    
    while True:
        user_query = input("Enter a proposal request:\n> ")
        
        # Check for exit commands
        cmd = user_query.lower().strip()
        
        if cmd in ['exit', 'quit', 'q', 'bye']:
            print("👋 Goodbye!")
            break
        
        # Check if user wants to see only retrieved chunks
        show_retrieval_only = False
        if cmd.startswith('retrieve only'):
            show_retrieval_only = True
            user_query = user_query[len('retrieve only'):].strip()
            print("\nShowing only retrieved information for: " + user_query)
        else:
            print("\nGenerating proposal for: " + user_query)
        
        generate_proposal(user_query, show_retrieval_only)
        print("\n---\n")

def print_banner():
    """Print a welcome banner with information"""
    print("\n" + "=" * 60)
    print("🤖 RAGbot - AI Proposal Assistant")
    print("=" * 60)
    print("Supports .txt and .pdf files in the Documents/ directory")
    print("New Feature: Now shows retrieved information before generating proposals")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Register cleanup function to run on exit
    atexit.register(cleanup)
    
    # Initialize components
    initialize()
    
    print_banner()
    
    if len(sys.argv) > 1:
        # Non-interactive mode with command line argument
        query = " ".join(sys.argv[1:])
        
        # Check if first argument is "retrieve-only"
        show_retrieval_only = False
        if sys.argv[1].lower() == "retrieve-only" and len(sys.argv) > 2:
            show_retrieval_only = True
            query = " ".join(sys.argv[2:])
            print(f"\nRetrieving information for: {query}")
        
        generate_proposal(query, show_retrieval_only)
    else:
        # Interactive mode
        run_interactive()