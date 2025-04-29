import os
import openai
import re
from dotenv import load_dotenv
import time
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt, max_tokens=2048, model="gpt-4o", temperature=0.3, preserve_formatting=True):
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
            print(f"Error generating response (attempt {retry+1}/{max_retries}): {str(e)}")
            
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