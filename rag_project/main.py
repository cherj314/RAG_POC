from dotenv import load_dotenv
from rag.retriever import search_postgres
from rag.prompt_builder import build_prompt
from rag.generator import generate_response

# Load environment variables
load_dotenv()

def generate_proposal(user_input):
    # Step 1: Retrieve relevant chunks based on the input
    k = 5
    similarity_threshold = 0.5
    chunks = search_postgres(user_input, k=k, similarity_threshold=similarity_threshold)

    # Step 2: Check if relevant content was found
    if not chunks:
        print("No relevant content found. Please try again with a different query.")
        return
    
    # Step 3: Build a prompt using the retrieved chunks and the user input
    prompt = build_prompt(chunks, user_input)
    
    # Step 4: Print the retrieved chunks and prompt
    print("\n🔍 Retrieved Chunks:\n")
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        print(f"[{i}] {doc[:300]}... (Similarity Score: {score})\n")
    
    print("\n🧠 Final Prompt:\n")
    print(prompt)
    
    # Step 5: Generate response using llama-cli
    print("No LLM connected, update code to use.")
    # print("\n🤖 Generating response...\n")
    # response = generate_response(prompt)
    # print("\n✅ Response:\n")
    # print(response)
    # return response

if __name__ == "__main__":
    user_query = input("Enter a proposal request:\n> ")
    print("\nGenerating proposal...\n")
    generate_proposal(user_query)