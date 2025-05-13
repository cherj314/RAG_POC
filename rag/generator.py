import os, openai, re, requests, time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Generate a response using either OpenAI's or Ollama's models
def generate_response(
    prompt, 
    max_tokens=2048, 
    model_type=None,
    model=None, 
    temperature=0.3, 
    preserve_formatting=True
):
    if model_type is None:
        model_type = os.getenv("DEFAULT_MODEL_TYPE", "openai").lower()
    
    max_retries = 3
    backoff_factor = 1.5
    
    if model is None:
        if model_type == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif model_type == "ollama":
            model = os.getenv("OLLAMA_MODEL", "tinyllama")
    
    # Prepare system message for Harry Potter context
    system_message = "You are an expert on Harry Potter books with comprehensive knowledge of the series. " \
    "You are a helpful assistant and literary expert. Answer the question based EXCLUSIVELY on the provided passages from the books. " \
    "For each statement in your answer, cite the specific passage number in brackets, like [PASSAGE 2]. " \
    "If the information isn't in the provided passages, admit this clearly instead of making up information. " \
    "If directly quoting the text, use quotation marks and cite the passage. " \
    "Your answers should be comprehensive yet concise, focused on directly addressing the question with evidence from the text."
    
    for retry in range(max_retries):
        try:
            # Call the appropriate model service
            if model_type == "openai":
                response_content = _call_openai_api(prompt, system_message, model, max_tokens, temperature)
            elif model_type == "ollama":
                response_content = _call_ollama_api(prompt, system_message, model, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # If we need to preserve formatting
            if preserve_formatting:
                return format_response(response_content)
            else:
                return response_content
            
        except (openai.RateLimitError, requests.exceptions.RequestException) as e:
            # Handle rate limiting or network errors with exponential backoff
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                print(f"Request failed. Retrying in {sleep_time:.1f}s... ({retry+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                error_type = "Rate limit exceeded" if isinstance(e, openai.RateLimitError) else "Connection error"
                return f"Error: {error_type}. Please try your query again in a moment."
                
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
            print(f"Error generating response (attempt {retry+1}/{max_retries}): {str(e)}")
            
            # For critical errors, retry with simpler prompt on the last attempt
            if retry == max_retries - 1:
                try:
                    # Try one last time with a simpler prompt
                    simplified_prompt = "Answer the following Harry Potter question as concisely as possible:\n\n" + prompt.split("QUESTION:")[-1]
                    simplified_system = "You are a Harry Potter expert. Keep your response brief and direct."
                    
                    if model_type == "openai":
                        return _call_openai_api(simplified_prompt, simplified_system, model, max_tokens // 2, 0.2)
                    elif model_type == "ollama":
                        return _call_ollama_api(simplified_prompt, simplified_system, model, max_tokens // 2, 0.2)
                except:
                    return f"Error generating response. Please try rephrasing your question."
    
    # If we've exhausted all retries
    return "Error generating response after multiple attempts. Please try again later."

# Call OpenAI API
def _call_openai_api(prompt, system_message, model, max_tokens, temperature):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response.choices[0].message.content.strip()

# Call Ollama API
def _call_ollama_api(prompt, system_message, model, max_tokens, temperature):
    # Format the request for Ollama API
    # Create system and user messages
    messages = [
        {"role": "system", "content": system_message},
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
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    # Extract the response content
    result = response.json()
    return result.get("message", {}).get("content", "")

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
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            ollama_models = response.json().get("models", [])
            if ollama_models:
                models["ollama"] = [model["name"] for model in ollama_models]
    except:
        pass  # Use default Ollama models if we can't fetch them
    
    return models