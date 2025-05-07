import os, openai, re, requests, time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

# Generate a response using either OpenAI's or Ollama's models
def generate_response(
    prompt, 
    max_tokens=2048, 
    model_type=None,
    model=None, 
    temperature=0.3, 
    preserve_formatting=True
):
    # Determine model type if not specified
    if model_type is None:
        model_type = os.getenv("DEFAULT_MODEL_TYPE", "openai").lower()
    
    # Call appropriate generation function based on model type
    if model_type == "openai":
        return generate_with_openai(prompt, max_tokens, model, temperature, preserve_formatting)
    elif model_type == "ollama":
        return generate_with_ollama(prompt, max_tokens, model, temperature, preserve_formatting)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Generate a response using OpenAI's models
def generate_with_openai(
    prompt, 
    max_tokens=2048, 
    model=None, 
    temperature=0.3, 
    preserve_formatting=True
):
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    max_retries = 3
    backoff_factor = 1.5
    
    for retry in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert on Harry Potter books with comprehensive knowledge of the series."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # If we need to preserve formatting
            if preserve_formatting:
                return format_response(raw_content)
            else:
                return raw_content
            
        except openai.RateLimitError:
            # Handle rate limiting with exponential backoff
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                print(f"Rate limit reached. Retrying in {sleep_time:.1f}s... ({retry+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                return "Error: Rate limit exceeded. Please try your query again in a moment."
                
        except openai.APITimeoutError:
            # Handle timeout
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                print(f"Request timed out. Retrying in {sleep_time:.1f}s... ({retry+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                return "Error: The request timed out. Please try your query again."
                
        except Exception as e:
            # Log detailed error
            print(f"Error generating response with OpenAI (attempt {retry+1}/{max_retries}): {str(e)}")
            
            # For critical errors, retry with simpler prompt
            if retry == max_retries - 1:
                try:
                    # Try one last time with a simpler prompt
                    simplified_prompt = "Answer the following Harry Potter question as concisely as possible:\n\n" + prompt.split("QUESTION:")[-1]
                    
                    response = openai.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a Harry Potter expert. Keep your response brief and direct."},
                            {"role": "user", "content": simplified_prompt}
                        ],
                        max_tokens=max_tokens // 2,  # Reduce tokens for fallback
                        temperature=0.2,  # Lower temperature for more reliable response
                    )
                    
                    return response.choices[0].message.content.strip()
                except:
                    return f"Error generating response. Please try rephrasing your question."
    
    # If we've exhausted all retries
    return "Error generating response after multiple attempts. Please try again later."

# Generate a response using Ollama's models
def generate_with_ollama(
    prompt, 
    max_tokens=2048, 
    model=None, 
    temperature=0.3, 
    preserve_formatting=True
):
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "tinyllama")
    
    max_retries = 3
    backoff_factor = 1.5
    
    # Format the request for Ollama API
    # Create system and user messages
    messages = [
        {"role": "system", "content": "You are an expert on Harry Potter books with comprehensive knowledge of the series."},
        {"role": "user", "content": prompt}
    ]
    
    # Build the request payload
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    for retry in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            raw_content = result.get("message", {}).get("content", "")
            
            # Format if needed
            if preserve_formatting:
                return format_response(raw_content)
            else:
                return raw_content
                
        except requests.exceptions.RequestException as e:
            # Handle network errors
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                print(f"Ollama request failed. Retrying in {sleep_time:.1f}s... ({retry+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                print(f"Error connecting to Ollama: {str(e)}")
                return "Error: Could not connect to Ollama service. Please check if the service is running."
                
        except Exception as e:
            # Handle other errors
            print(f"Error generating response with Ollama (attempt {retry+1}/{max_retries}): {str(e)}")
            
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                time.sleep(sleep_time)
            else:
                # Final attempt with simplified prompt
                try:
                    # Simplify the prompt for the last attempt
                    simplified_prompt = "Answer this Harry Potter question briefly: " + prompt.split("QUESTION:")[-1]
                    
                    simple_payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a Harry Potter expert. Keep answers brief."},
                            {"role": "user", "content": simplified_prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": max_tokens // 2
                        }
                    }
                    
                    response = requests.post(url, json=simple_payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    return result.get("message", {}).get("content", "")
                    
                except:
                    return "Error generating response with Ollama. Please try again later."
    
    # If we've exhausted all retries
    return "Error generating response after multiple attempts. Please try again later."

# Format the response to ensure readability and proper formatting
def format_response(raw_content):
    # Replace any instances of 3+ newlines with exactly 2 newlines
    formatted_content = re.sub(r'\n{3,}', '\n\n', raw_content)
    
    # Add newlines before headers if they don't already have them
    formatted_content = re.sub(r'(?<!\n\n)(#{1,6}\s)', r'\n\n\1', formatted_content)
    
    # Ensure all bullet points have proper spacing
    formatted_content = re.sub(r'(?<!\n)(\s*[-*+]\s)', r'\n\1', formatted_content)
    
    # Preserve citation brackets
    formatted_content = re.sub(r'\[PASSAGE\s+(\d+)\]', r'[PASSAGE \1]', formatted_content)
    
    return formatted_content

# Get the list of available models for OpenAI and Ollama
def get_available_models():
    models = {
        "openai": ["gpt-4o", "gpt-3.5-turbo"],
        "ollama": ["tinyllama"]
    }
    
    # Try to get available Ollama models
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            ollama_models = response.json().get("models", [])
            if ollama_models:
                models["ollama"] = [model["name"] for model in ollama_models]
    except:
        pass  # Use default Ollama models if we can't fetch them
    
    return models