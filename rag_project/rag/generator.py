import os
import subprocess
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get llama cli and model paths from environment
LLAMA_CLI_PATH = os.getenv("LLAMA_CLI_PATH", "llama-cli.exe")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "path/to/model.gguf")

def generate_response(prompt, max_tokens=512):
    """
    Generate a response using llama-cli.exe from the command line
    """
    try:
        # Create a command to run llama-cli with the prompt
        cmd = [
            LLAMA_CLI_PATH,
            "-m", LLAMA_MODEL_PATH,
            "-n", str(max_tokens),  # Number of tokens to generate
            "-p", prompt  # The prompt
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # The output will include the prompt as well, so we need to extract just the response
        # This is a simplistic way to separate the response - you might need to adjust this
        output = result.stdout
        response = output.split(prompt, 1)[-1].strip()
        
        return response
    
    except subprocess.CalledProcessError as e:
        print(f"Error running llama-cli: {e}")
        print(f"stderr: {e.stderr}")
        return f"Error generating response: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error generating response: {str(e)}"