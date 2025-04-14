from rag.retriever import search_postgres
from rag.prompt_builder import build_prompt
# from rag.generator import generate_response  # Uncomment when model is connected

def generate_proposal(user_input):
    # Step 1: Retrieve relevant chunks based on the input
    # Set k (number of results) and similarity threshold (e.g., 0.7)
    k = 5
    similarity_threshold = 0.5
    chunks = search_postgres(user_input, k=k, similarity_threshold=similarity_threshold)

    # Step 2: Check if relevant content was found
    if not chunks:
        print("No relevant content found. Please try again with a different query.")
        return
    
    # Step 3: Build a prompt using the retrieved chunks and the user input
    prompt = build_prompt(chunks, user_input)
    
    # Step 4: For now, we will print the prompt instead of generating a response
    print("\n🔍 Retrieved Chunks:\n")
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        print(f"[{i}] {doc[:300]}... (Similarity Score: {score})\n")  # Show preview of each chunk with score
    
    print("\n🧠 Final Prompt:\n")
    print(prompt)
    
    # Step 5: Model generation will happen here once you're ready
    # response = generate_response(prompt)
    # return response

if __name__ == "__main__":
    user_query = input("Enter a proposal request:\n> ")
    print("\nGenerating proposal...\n")
    generate_proposal(user_query)
