import sys
import os
import uvicorn
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from rag.retriever import search_postgres, get_embed_model
from rag.prompt_builder import build_prompt
from rag.generator import generate_response
from rag.config import init_db_pool, get_db_connection, release_connection

# Initialize FastAPI app
app = FastAPI(title="RAGbot API", description="API for the RAG-based proposal generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global flag to track initialization status
is_initialized = False

# Define request and response models
class ProposalRequest(BaseModel):
    query: str
    similarity_threshold: Optional[float] = 0.5
    max_chunks: Optional[int] = 5
    show_retrieved_only: Optional[bool] = False

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    model: Optional[str] = "ragbot"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    show_retrieved: Optional[bool] = True
    
class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None
    score: float

# Initialize database pool on startup
@app.on_event("startup")
async def startup_event():
    global is_initialized
    try:
        init_db_pool()
        get_embed_model()
        is_initialized = True
    except Exception as e:
        print(f"Error during startup: {str(e)}")

# Ensure components are initialized
def ensure_initialized():
    """Ensure all required components are initialized"""
    global is_initialized
    
    if not is_initialized:
        try:
            init_db_pool()
            get_embed_model()
            is_initialized = True
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to initialize components")

# Format retrieved chunks for display
def format_retrieved_chunks(chunks):
    """Format retrieved chunks for display to the user"""
    formatted_chunks = []
    for i, (doc, metadata, score) in enumerate(chunks, 1):
        source = metadata.get("file_name", "Unknown") if metadata else "Unknown"
        preview = doc[:300] + "..." if len(doc) > 300 else doc
        formatted_chunks.append({
            "content": preview,
            "source": source,
            "score": score
        })
    return formatted_chunks

# Routes
@app.post("/api/generate-proposal")
async def generate_proposal(request: ProposalRequest):
    try:
        ensure_initialized()
        
        # Retrieve relevant chunks
        chunks = search_postgres(
            request.query,
            k=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        if not chunks:
            return {
                "success": False,
                "message": "No relevant content found for the query."
            }
        
        # Format retrieved chunks for response
        retrieved_chunks = format_retrieved_chunks(chunks)
        
        # If user only wants to see retrieved chunks, return them without generating
        if request.show_retrieved_only:
            return {
                "success": True,
                "retrieved_chunks": retrieved_chunks,
                "metadata": {"query": request.query}
            }
        
        # Build prompt and generate response
        prompt = build_prompt(chunks, request.query)
        start_time = time.time()
        response = generate_response(prompt)
        
        return {
            "success": True,
            "proposal": response,
            "retrieved_chunks": retrieved_chunks,
            "metadata": {
                "generation_time": f"{time.time() - start_time:.2f}s",
                "query": request.query
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

@app.post("/chat/completions")
async def chat_completions_no_prefix(request: Request):
    """OpenAI-compatible chat completions endpoint (without /api prefix)"""
    # Parse the request body manually to avoid validation errors
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
        return await process_chat_completion(chat_request)
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return format_error_response(str(e))

@app.post("/api/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint for OpenWebUI integration"""
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
        return await process_chat_completion(chat_request)
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return format_error_response(str(e))

# Update the process_chat_completion function to support model selection

async def process_chat_completion(request: ChatRequest):
    """Process a chat completion request with fixed model selection"""
    try:
        ensure_initialized()
        
        # Extract the last user message as our query
        last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        
        if not last_user_message:
            return format_error_response("No user message found in the request")
        
        # Process the query using our RAG pipeline
        chunks = search_postgres(
            last_user_message,
            k=5,
            similarity_threshold=0.3
        )
        
        # Format retrieved chunks
        retrieved_chunks = format_retrieved_chunks(chunks)
        show_retrieved = getattr(request, 'show_retrieved', True)
        
        # Get model information
        requested_model = request.model or "ragbot"
        temperature = request.temperature or 0.7
        max_tokens = request.max_tokens or 2048
        
        # Import module inside function to avoid circular imports
        from rag.generator import generate_response, MODEL_TYPE, MODEL_NAME
        
        # Use the default model from environment variables
        # This fixes the issue where "model" was being used instead of "tinyllama"
        model_name = os.getenv("OLLAMA_MODEL", "tinyllama") if MODEL_TYPE == "ollama" else MODEL_NAME
        
        print(f"Using model: {model_name} (type: {MODEL_TYPE})")
        
        # Generate text based on retrieved chunks
        if chunks:
            prompt = build_prompt(chunks, last_user_message)
            
            if show_retrieved:
                # Show retrieved chunks + generated response
                retrieved_content = "📚 **Retrieved Relevant Information:**\n\n"
                for i, chunk_info in enumerate(retrieved_chunks, 1):
                    retrieved_content += f"**[{i}] Source: {chunk_info['source']} (Score: {chunk_info['score']:.3f})**\n"
                    chunk_text = chunk_info['content']
                    if not chunk_text.endswith('\n'):
                        chunk_text += '\n'
                    retrieved_content += f"{chunk_text}\n"
                retrieved_content += "---\n\n**Generating proposal based on these sources...**\n\n"
                
                # Generate the response
                generated_response = generate_response(
                    prompt=prompt, 
                    max_tokens=max_tokens, 
                    model=model_name, 
                    temperature=temperature,
                    preserve_formatting=True
                )
                
                generated_text = retrieved_content + generated_response
            else:
                # Just generate the response without showing chunks
                generated_text = generate_response(
                    prompt=prompt, 
                    max_tokens=max_tokens, 
                    model=model_name, 
                    temperature=temperature,
                    preserve_formatting=True
                )
        else:
            # No relevant chunks found
            generated_text = (
                "❌ **No relevant information found in my knowledge base.**\n\n"
                "I'm sorry, but I couldn't find any relevant information in my knowledge base "
                "about your query. Could you please rephrase your question or ask about something "
                "related to Harry Potter, which is my area of expertise?"
            ) if show_retrieved else (
                "I'm sorry, but I couldn't find any relevant information in my knowledge base "
                "about your query. Could you please rephrase your question or ask about something "
                "related to Harry Potter, which is my area of expertise?"
            )
        
        # Format as OpenAI API response
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(generated_text.split())
            }
        }
        
        # If streaming is requested
        if request.stream:
            return StreamingResponse(stream_response(generated_text))
        
        return response
    
    except Exception as e:
        print(f"Error in process_chat_completion: {str(e)}")
        import traceback
        traceback.print_exc()
        return format_error_response(f"I encountered an error processing your request: {str(e)}")

def format_error_response(error_message):
    """Format a proper error response in OpenAI format"""
    return {
        "id": f"chatcmpl-error-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "ragbot",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"I encountered an error processing your request: {error_message}. The RAGbot system administrators have been notified."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

# Endpoint to get only retrieved chunks
@app.post("/api/retrieve-chunks")
async def retrieve_chunks(request: ProposalRequest):
    try:
        ensure_initialized()
        chunks = search_postgres(
            request.query,
            k=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        if not chunks:
            return {
                "success": False,
                "message": "No relevant content found for the query."
            }
        
        return {
            "success": True,
            "retrieved_chunks": format_retrieved_chunks(chunks),
            "metadata": {"query": request.query}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

async def stream_response(content: str):
    """Generate streaming response chunks for OpenAI-compatible streaming"""
    # Check if content starts with retrieved chunks section
    retrieved_section_end = content.find("---\n\n**Generating proposal")
    
    # If there's a retrieved section, stream it as a block first
    if retrieved_section_end > 0:
        retrieved_content = content[:retrieved_section_end + 4]  # Include the "---\n\n"
        proposal_content = content[retrieved_section_end + 4:]
        
        # Stream the retrieved content as a block
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "ragbot",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": retrieved_content},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.1)  # Pause after retrieved content
        
        # Update content to be just the proposal part
        content = proposal_content
    
    # Split the content by paragraphs
    paragraphs = content.split('\n\n')
    
    for p_idx, paragraph in enumerate(paragraphs):
        # For very long paragraphs, split them further
        if len(paragraph) > 300:
            # Split into sentences
            sentences = paragraph.replace('\n', ' ').split('. ')
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                if not sentence.endswith('.'):
                    sentence += '.'
                
                sentence_length = len(sentence)
                
                # If this sentence would make the chunk too long, send the current chunk
                if current_length + sentence_length > 200 and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "ragbot",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text + " "},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.05)
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Send any remaining content in the current chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ragbot",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.05)
        else:
            # Short paragraph - send as is
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "ragbot",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": paragraph},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.05)
        
        # Add paragraph breaks between paragraphs, but not after the last one
        if p_idx < len(paragraphs) - 1:
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "ragbot",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "\n\n"},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.02)  # Shorter pause for paragraph breaks
    
    # End the stream
    data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "ragbot",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/models")
async def list_models_no_prefix():
    """OpenAI-compatible models endpoint (without /api prefix)"""
    return await list_models()

# Update the models endpoint in api.py to include Ollama models
# Replace the existing list_models functions with this code

# Replace the list_models functions in api.py with this simplified version

@app.get("/models")
async def list_models_no_prefix():
    """OpenAI-compatible models endpoint (without /api prefix)"""
    return await list_models()

@app.get("/api/models")
async def list_models():
    """OpenAI-compatible models endpoint that includes both OpenAI and Ollama models"""
    try:
        from rag.generator import list_available_models, MODEL_TYPE
        
        # Get available models
        models_list = list_available_models()
        
        # Convert to OpenAI-compatible format
        openai_models = []
        for model in models_list:
            model_id = model["id"]
            model_type = model["type"]
            
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization-owner" if model_type == "openai" else "ollama",
                "permission": [],
                "root": model_type,
                "parent": None
            })
        
        # Always include the default "ragbot" model for backward compatibility
        openai_models.append({
            "id": "ragbot",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "organization-owner",
            "permission": [],
            "root": "ragbot",
            "parent": None
        })
        
        return {
            "object": "list",
            "data": openai_models
        }
        
    except Exception as e:
        print(f"Error in list_models: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback response with just the ragbot model
        return {
            "object": "list",
            "data": [
                {
                    "id": "ragbot",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "organization-owner",
                    "permission": [],
                    "root": "ragbot",
                    "parent": None
                }
            ]
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    from rag.config import DB_POOL
    import os
    
    # Get model configuration
    model_type = os.getenv("MODEL_TYPE", "openai")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    
    health_status = {
        "status": "ok" if is_initialized else "initializing",
        "database": "connected" if DB_POOL is not None else "disconnected",
        "embedding_model": "loaded" if get_embed_model() is not None else "not_loaded",
        "timestamp": int(time.time()),
        "llm_config": {
            "model_type": model_type,
            "model_name": model_name
        }
    }
    
    # Test database connection
    if DB_POOL is not None:
        try:
            conn = get_db_connection()
            if conn:
                health_status["database_test"] = "success"
                release_connection(conn)
            else:
                health_status["database_test"] = "failed"
        except Exception as e:
            health_status["database_test"] = f"error: {str(e)}"
    
    # Simple check for Ollama if configured to use it
    if model_type == "ollama":
        try:
            import requests
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            try:
                response = requests.get(f"{ollama_base_url}/api/version", timeout=2)
                health_status["ollama"] = {
                    "status": "ok" if response.status_code == 200 else "error",
                    "details": "Connected to Ollama API" if response.status_code == 200 else f"HTTP {response.status_code}"
                }
            except Exception as e:
                health_status["ollama"] = {
                    "status": "error",
                    "details": f"Connection error: {str(e)}"
                }
        except ImportError:
            health_status["ollama"] = {
                "status": "error",
                "details": "Requests module not available"
            }
    
    return health_status

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)