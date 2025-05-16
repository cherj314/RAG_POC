import os, re, requests, time
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

# Generate a response using either OpenAI or Ollama models
def generate_response(prompt, max_tokens=2048, model_type=None, model=None, temperature=0.3, preserve_formatting=True):
    model_type = model_type or os.getenv("DEFAULT_MODEL_TYPE", "openai").lower()
    model = model or (os.getenv("OPENAI_MODEL", "gpt-4o") if model_type == "openai" else os.getenv("OLLAMA_MODEL", "tinyllama"))
    
    # System message for Harry Potter context
    system_message = "You are an expert on Harry Potter books with comprehensive knowledge of the series. Answer the question based EXCLUSIVELY on the provided passages from the books. For each statement in your answer, cite the specific passage number in brackets, like [PASSAGE 2]. If the information isn't in the provided passages, admit this clearly instead of making up information. If directly quoting the text, use quotation marks and cite the passage. Your answers should be comprehensive yet concise, focusing on directly addressing the question with evidence from the text."
    
    # Retry logic
    max_retries, backoff_factor = 3, 1.5
    
    for retry in range(max_retries):
        try:
            # Call the appropriate API
            if model_type == "openai":
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content.strip()
            elif model_type == "ollama":
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Format response if needed
            return format_response(content) if preserve_formatting else content
            
        except (openai.RateLimitError, requests.exceptions.RequestException) as e:
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                print(f"Request failed. Retrying in {sleep_time:.1f}s... ({retry+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                error_type = "Rate limit exceeded" if isinstance(e, openai.RateLimitError) else "Connection error"
                return f"Error: {error_type}. Please try your query again in a moment."
        except openai.APITimeoutError:
            if retry < max_retries - 1:
                sleep_time = backoff_factor ** retry
                print(f"Request timed out. Retrying in {sleep_time:.1f}s... ({retry+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                return "Error: The request timed out. Please try your query again."
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            # Try one last time with a simpler prompt on the last attempt
            if retry == max_retries - 1:
                try:
                    simplified_prompt = "Answer the following Harry Potter question as concisely as possible:\n\n" + prompt.split("QUESTION:")[-1]
                    simplified_system = "You are a Harry Potter expert. Keep your response brief and direct."
                    
                    if model_type == "openai":
                        return openai.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": simplified_system},
                                {"role": "user", "content": simplified_prompt}
                            ],
                            max_tokens=max_tokens // 2,
                            temperature=0.2,
                        ).choices[0].message.content.strip()
                    elif model_type == "ollama":
                        return requests.post(
                            f"{OLLAMA_BASE_URL}/api/chat",
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": simplified_system},
                                    {"role": "user", "content": simplified_prompt}
                                ],
                                "stream": False,
                                "options": {"temperature": 0.2, "num_predict": max_tokens // 2}
                            }
                        ).json().get("message", {}).get("content", "")
                except:
                    return f"Error generating response. Please try rephrasing your question."
    
    return "Error generating response after multiple attempts. Please try again later."

def format_response(raw_content):
    """Format the response to ensure readability and proper formatting"""
    # Replace multiple newlines with double newlines
    content = re.sub(r'\n{3,}', '\n\n', raw_content)
    # Add newlines before headers
    content = re.sub(r'(?<!\n\n)(#{1,6}\s)', r'\n\n\1', content)
    # Ensure proper spacing for bullet points
    content = re.sub(r'(?<!\n)(\s*[-*+]\s)', r'\n\1', content)
    # Preserve citation brackets
    content = re.sub(r'\[PASSAGE\s+(\d+)\]', r'[PASSAGE \1]', content)
    return content

def get_available_models():
    """Get the list of available models for OpenAI and Ollama"""
    models = {
        "openai": ["gpt-4o"],
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

def build_prompt(context_chunks, user_prompt):
    """Build a prompt that combines retrieved context with the user request"""
    # Format chunks with metadata
    formatted_chunks = []
    
    for i, (doc, metadata, score) in enumerate(context_chunks, 1):
        # Clean document text
        clean_doc = doc.strip()
        
        # Extract metadata
        source = metadata.get('file_name', '') if metadata and isinstance(metadata, dict) else ''
        page_info = f", Page {metadata.get('page_num', '')}" if metadata and metadata.get('page_num') else ''
        section_info = f", Section: {metadata.get('current_section', '')}" if metadata and metadata.get('current_section') else ''
        
        # Create metadata string
        meta_str = f"[Source: {source}{page_info}{section_info}]" if source else ""
        
        formatted_chunks.append(f"PASSAGE {i} (RELEVANCE: {score:.2f}) {meta_str}:\n{clean_doc}")
    
    # Join chunks with clear separation
    context = "\n\n---\n\n".join(formatted_chunks)
    
    # Create the optimized Harry Potter prompt
    return f"RETRIEVED PASSAGES:\n\n{context}\n\nQUESTION: {user_prompt}\n\nANSWER:"