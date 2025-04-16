from dotenv import load_dotenv
from rag.retriever import search_postgres
from rag.prompt_builder import build_prompt
from rag.generator import generate_response

# Load environment variables
load_dotenv()

def generate_proposal(user_input):
    """
    Generate a proposal based on user input using RAG methodology.
    
    Args:
        user_input (str): The user's request for a proposal
        
    Returns:
        str or None: Generated proposal text if successful, None otherwise
    """
    # Step 1: Retrieve relevant chunks using vector similarity
    k = 5  # Number of chunks to retrieve
    similarity_threshold = 0.5  # Minimum similarity score
    chunks = search_postgres(user_input, k=k, similarity_threshold=similarity_threshold)

    # Step 2: Handle case where no relevant content found
    if not chunks:
        print("No relevant content found. Please try again with a different query.")
        return None
    
    # Step 3: Build prompt with retrieved context and user input
    prompt = build_prompt(chunks, user_input)
    
    # Step 4: Display retrieved information for transparency
    print("\n🔍 Retrieved Chunks:\n")
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        print(f"[{i}] {doc[:300]}... (Similarity Score: {score:.4f})\n")
    
    print("\n🧠 Final Prompt:\n")
    print(prompt)
    
    # Step 5: Generate response (uncomment when ready to use an LLM)
    # response = generate_response(prompt)
    # print("\n✅ Response:\n")
    # print(response)
    # return response
    
    # Temporary placeholder until LLM integration is complete
    print("\nLLM integration is currently disabled. To enable generation:")
    print("1. Configure your LLM in .env")
    print("2. Uncomment the generation code in main.py")
    return None

if __name__ == "__main__":
    user_query = input("Enter a proposal request:\n> ")
    print("\nGenerating proposal...\n")
    generate_proposal(user_query)