import os
import time
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client
openai.api_key = OPENAI_API_KEY

def generate_response(prompt, max_tokens=2048, model="gpt-4o"):
    """
    Generate a response using OpenAI's GPT-4o model.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        model (str): The OpenAI model to use
        
    Returns:
        str: The generated response
    """
    # Provide an estimate of generation time
    print(f"⏱️ Estimated generation time: 5-15 seconds (depending on API load)")
    
    try:
        # Start timing
        start_time = time.time()
        
        print("🧠 Starting GPT-4o generation...")
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert software proposal writer for a professional software development company."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        # Extract response content
        generated_text = response.choices[0].message.content.strip()
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed = end_time - start_time
            
        print(f"✅ Generation completed in {elapsed:.2f}s")
        
        return generated_text
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}")
        return f"Error generating response: {str(e)}"