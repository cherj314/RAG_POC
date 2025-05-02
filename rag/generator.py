import os
import openai
import re
import requests
from dotenv import load_dotenv
import time
from typing import Dict, Any, Optional, List

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "openai").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

def generate_response(prompt, max_tokens=2048, model=None, temperature=0.3, preserve_formatting=True):
    """
    Generate a response using either OpenAI or Ollama based on MODEL_TYPE setting.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        model (str, optional): Override the default model
        temperature (float): Temperature for generation (higher is more creative)
        preserve_formatting (bool): Whether to preserve paragraph breaks and formatting
        
    Returns:
        str: The generated response
    """
    # Use environment variables or defaults if not provided
    model_type = MODEL_TYPE
    model_name = model or MODEL_NAME
    
    print(f"Generating response with: model_type={model_type}, model={model_name}")
    
    # Generate response based on model type
    if model_type == "openai":
        return generate_response_with_openai(
            prompt=prompt,
            max_tokens=max_tokens,
            model=model_name,
            temperature=temperature,
            preserve_formatting=preserve_formatting
        )
    elif model_type == "ollama":
        return generate_response_with_ollama(
            prompt=prompt,
            max_tokens=max_tokens,
            model=model_name,
            temperature=temperature,
            preserve_formatting=preserve_formatting
        )
    else:
        return f"Error: Unsupported model type '{model_type}'. Please check your configuration."

def generate_response_with_openai(prompt, max_tokens=2048, model="gpt-4o", temperature=0.3, preserve_formatting=True):
    """
    Generate a response using OpenAI's model with enhanced error handling and retries.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        model (str): The OpenAI model to use
        temperature (float): Temperature for generation (higher is more creative)
        preserve_formatting (bool): Whether to preserve paragraph breaks and formatting
        
    Returns:
        str: The generated response
    """
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
                # Replace any instances of 3+ newlines with exactly 2 newlines
                formatted_content = re.sub(r'\n{3,}', '\n\n', raw_content)
                
                # Add newlines before headers if they don't already have them
                formatted_content = re.sub(r'(?<!\n\n)(#{1,6}\s)', r'\n\n\1', formatted_content)
                
                # Ensure all bullet points have proper spacing
                formatted_content = re.sub(r'(?<!\n)(\s*[-*+]\s)', r'\n\1', formatted_content)
                
                # Preserve citation brackets
                formatted_content = re.sub(r'\[PASSAGE\s+(\d+)\]', r'[PASSAGE \1]', formatted_content)
                
                return formatted_content
            else:
                return raw_content
            
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

def generate_response_with_ollama(prompt, max_tokens=2048, model="tinyllama", temperature=0.3, preserve_formatting=True):
    """
    Generate a response using Ollama API.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        model (str): The Ollama model to use (e.g., tinyllama)
        temperature (float): Temperature for generation
        preserve_formatting (bool): Whether to preserve paragraph breaks and formatting
        
    Returns:
        str: The generated response
    """
    try:
        # Extract just the question if the prompt has the RAG format
        if "QUESTION:" in prompt:
            question = prompt.split("QUESTION:", 1)[1].strip()
        else:
            question = prompt
        
        # Simplify the system prompt
        system_prompt = "You are an expert on Harry Potter books with comprehensive knowledge of the series."
        
        # Prepare the payload for Ollama API
        payload = {
            "model": model,
            "prompt": question,
            "system": system_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False
        }
        
        # Make the API request to Ollama
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            result = data.get("response", "").strip()
            return result
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            print(error_msg)
            return f"Error: {error_msg}"
                
    except Exception as e:
        error_msg = f"Error generating response with Ollama: {str(e)}"
        print(error_msg)
        return f"Error: {error_msg}"

def check_ollama_health():
    """
    Check if the Ollama service is healthy.
    
    Returns:
        dict: Status information about Ollama
    """
    try:
        # Check if Ollama service is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        
        if response.status_code == 200:
            return {
                "status": "ok",
                "details": response.json()
            }
        else:
            return {
                "status": "error",
                "message": f"Ollama returned status code {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to Ollama: {str(e)}"
        }

def list_available_models():
    """
    Get a list of all available models based on configured model types.
    
    Returns:
        list: Available model configurations
    """
    models = []
    
    # Always include OpenAI models if API key is set
    if os.getenv("OPENAI_API_KEY"):
        models.extend([
            {"id": "gpt-4o", "name": "GPT-4o", "type": "openai", "context_length": 128000},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "type": "openai", "context_length": 16000}
        ])
    
    # Check Ollama models
    try:
        # Check if Ollama service is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            available_models = data.get("models", [])
            
            # Add each available Ollama model
            for model_info in available_models:
                model_name = model_info.get("name", "")
                if model_name:
                    models.append({
                        "id": model_name,
                        "name": f"Ollama: {model_name}",
                        "type": "ollama",
                        "context_length": 4096  # Default context length for most small models
                    })
        
        # Always add the default Ollama model even if it's not yet downloaded
        default_ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama")
        if not any(model["id"] == default_ollama_model for model in models):
            models.append({
                "id": default_ollama_model,
                "name": f"Ollama: {default_ollama_model}",
                "type": "ollama",
                "context_length": 4096
            })
            
    except Exception as e:
        print(f"Error checking Ollama models: {str(e)}")
        # Add a placeholder for the default Ollama model
        default_ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama")
        models.append({
            "id": default_ollama_model,
            "name": f"Ollama: {default_ollama_model} (Not connected)",
            "type": "ollama",
            "context_length": 4096
        })
    
    return models