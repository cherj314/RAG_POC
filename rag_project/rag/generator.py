from llama_cpp import Llama

# Adjust model path as needed
llm = Llama(model_path="path/to/model.gguf")

def generate_response(prompt, max_tokens=512):
    result = llm(prompt, max_tokens=max_tokens)
    return result['choices'][0]['text']
