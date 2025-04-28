import os
import openai
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt, max_tokens=2048, model="gpt-4o", preserve_formatting=True):
    """
    Generate a response using OpenAI's model.
    
    Args:
        prompt (str): The input prompt for the LLM
        max_tokens (int): Maximum number of tokens to generate
        model (str): The OpenAI model to use
        preserve_formatting (bool): Whether to preserve paragraph breaks and formatting
        
    Returns:
        str: The generated response
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert software proposal writer for a professional software development company."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
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
            
            return formatted_content
        else:
            return raw_content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"