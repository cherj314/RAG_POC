import os
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get llama CLI and model paths from environment
LLAMA_CLI_PATH = os.getenv("LLAMA_CLI_PATH", "llama-cli.exe")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "path/to/model.gguf")

def generate_response(prompt, max_tokens=512):
    """
    Generate a response using llama-cli from the command line.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated response
    """
    try:
        # Build command for llama-cli
        cmd = [
            LLAMA_CLI_PATH,
            "-m", LLAMA_MODEL_PATH,
            "-n", str(max_tokens),
            "-p", prompt
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract just the response (remove the prompt from output)
        output = result.stdout
        response = output.split(prompt, 1)[-1].strip()
        
        return response
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running llama-cli: {e}\nStderr: {e.stderr}"
        print(error_msg)
        return f"Error generating response: {str(e)}"
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(error_msg)
        return f"Error generating response: {str(e)}"