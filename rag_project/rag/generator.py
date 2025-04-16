import os
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get llama CLI and model paths from environment
LLAMA_CLI_PATH = os.getenv("LLAMA_CLI_PATH", "llama-cli.exe")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "path/to/model.gguf")

# Cache for generation times to provide better estimates
recent_token_rates = []

def estimate_generation_time(prompt_length, max_tokens=512):
    """
    Estimate how long generation will take based on recent history.
    
    Args:
        prompt_length: Length of the prompt in characters
        max_tokens: Maximum tokens to generate
        
    Returns:
        float: Estimated time in seconds
    """
    global recent_token_rates
    
    # If we don't have history, use a conservative estimate
    if not recent_token_rates:
        return max_tokens * 0.1  # Assume 10 tokens per second
    
    # Calculate average token generation rate from history
    avg_rate = sum(recent_token_rates) / len(recent_token_rates)
    return max_tokens / avg_rate

def generate_response(prompt, max_tokens=512):
    """
    Generate a response using llama-cli from the command line.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated response
    """
    global recent_token_rates
    
    # Provide an estimate of generation time
    est_time = estimate_generation_time(len(prompt), max_tokens)
    print(f"⏱️ Estimated generation time: {est_time:.1f} seconds")
    
    try:
        # Build command for llama-cli
        cmd = [
            LLAMA_CLI_PATH,
            "-m", LLAMA_MODEL_PATH,
            "-n", str(max_tokens),
            "--ctx_size", "4096",  # Larger context size for better understanding
            "-t", "8",  # Use 8 threads for faster inference
            "-p", prompt
        ]
        
        # Start timing
        start_time = time.time()
        
        # Run the command and capture output
        print("🧠 Starting LLM generation...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract just the response (remove the prompt from output)
        output = result.stdout
        response = output.split(prompt, 1)[-1].strip()
        
        # Calculate and record token rate for future estimates
        end_time = time.time()
        elapsed = end_time - start_time
        tokens_generated = len(response.split()) * 1.3  # Rough estimate of tokens
        token_rate = tokens_generated / elapsed
        
        # Keep only the last 5 generation rates for the estimate
        recent_token_rates.append(token_rate)
        if len(recent_token_rates) > 5:
            recent_token_rates.pop(0)
            
        print(f"✅ Generation completed in {elapsed:.2f}s ({token_rate:.1f} tokens/sec)")
        
        return response
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running llama-cli: {e}\nStderr: {e.stderr}"
        print(f"❌ {error_msg}")
        return f"Error generating response: {str(e)}"
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}")
        return f"Error generating response: {str(e)}"