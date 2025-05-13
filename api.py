import sys, os, uvicorn, time, json, asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from rag.retriever import search_postgres, get_embed_model
from rag.prompt_builder import build_prompt
from rag.generator import generate_response, get_available_models
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

# Get model configuration
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE").lower()
AVAILABLE_MODEL_TYPES = os.getenv("AVAILABLE_MODEL_TYPES").lower().split(",")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Define request and response models
class ProposalRequest(BaseModel):
    query: str
    similarity_threshold: Optional[float] = 0.5
    max_chunks: Optional[int] = 5
    show_retrieved_only: Optional[bool] = False
    model_type: Optional[str] = DEFAULT_MODEL_TYPE
    model_name: Optional[str] = None

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

# Parse model info from model string
def parse_model_info(model_string):
    if not model_string or model_string == "ragbot":
        return DEFAULT_MODEL_TYPE, None
    
    if "/" in model_string:
        parts = model_string.split("/", 1)
        model_type = parts[0].lower()
        model_name = parts[1] if len(parts) > 1 else None
        
        # Validate model type
        if model_type not in AVAILABLE_MODEL_TYPES:
            model_type = DEFAULT_MODEL_TYPE
            
        return model_type, model_name
    
    # If no slash, assume it's a model name for the default type
    return DEFAULT_MODEL_TYPE, model_string

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
        
        # Use specified model type and name
        model_type = request.model_type or DEFAULT_MODEL_TYPE
        model_name = request.model_name
        
        response = generate_response(
            prompt,
            model_type=model_type,
            model=model_name
        )
        
        return {
            "success": True,
            "proposal": response,
            "retrieved_chunks": retrieved_chunks,
            "metadata": {
                "generation_time": f"{time.time() - start_time:.2f}s",
                "query": request.query,
                "model_type": model_type,
                "model_name": model_name or "default"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

# OpenAI-compatible chat completions endpoint for OpenWebUI integration
@app.post("/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
        return await process_chat_completion(chat_request)
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return format_error_response(str(e))

# Process a chat completion request
async def process_chat_completion(request: ChatRequest):
    try:
        ensure_initialized()
        
        # Extract the last user message as our query
        last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        
        if not last_user_message:
            return format_error_response("No user message found in the request")
        
        # Parse model type and name from the model string
        model_type, model_name = parse_model_info(request.model)
        
        # Process the query using our RAG pipeline
        chunks = search_postgres(
            last_user_message,
            k=5,
            similarity_threshold=0.3
        )
        
        # Format retrieved chunks
        retrieved_chunks = format_retrieved_chunks(chunks)
        show_retrieved = getattr(request, 'show_retrieved', True)
        
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
                
                generated_response = generate_response(
                    prompt, 
                    max_tokens=request.max_tokens,
                    model_type=model_type,
                    model=model_name,
                    temperature=request.temperature,
                    preserve_formatting=True
                )
                generated_text = retrieved_content + generated_response
            else:
                # Just generate the response without showing chunks
                generated_text = generate_response(
                    prompt, 
                    max_tokens=request.max_tokens,
                    model_type=model_type,
                    model=model_name,
                    temperature=request.temperature,
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
            "model": request.model or "ragbot",
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

# Format error response in OpenAI format
def format_error_response(error_message):
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

# Streaming response generator for OpenAI-compatible streaming
async def stream_response(content: str):
    # Split the content into paragraphs
    paragraphs = content.split('\n\n')
    
    for i, paragraph in enumerate(paragraphs):
        # Create message data for the paragraph
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
        
        # Send the chunk
        yield f"data: {json.dumps(data)}\n\n"
        
        # Add paragraph breaks between paragraphs, but not after the last one
        if i < len(paragraphs) - 1:
            paragraph_break_data = {
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
            yield f"data: {json.dumps(paragraph_break_data)}\n\n"
        
        # Small delay between chunks
        await asyncio.sleep(0.05)
    
    # End the stream
    end_data = {
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
    yield f"data: {json.dumps(end_data)}\n\n"
    yield "data: [DONE]\n\n"

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

# OpenAI-compatible models endpoint
@app.get("/models")
@app.get("/api/models")
async def list_models():
    try:
        # Get available models
        available_models = get_available_models()
        
        model_data = [
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
        
        # Add OpenAI models
        for model in available_models.get("openai", []):
            model_data.append({
                "id": f"openai/{model}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
                "permission": [],
                "root": model,
                "parent": None
            })
        
        # Add Ollama models
        for model in available_models.get("ollama", []):
            model_data.append({
                "id": f"ollama/{model}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "permission": [],
                "root": model,
                "parent": None
            })
        
        return {
            "object": "list",
            "data": model_data
        }
    except Exception as e:
        print(f"Error listing models: {str(e)}")
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

# Health check endpoint
@app.get("/api/health")
async def health_check():
    from rag.config import DB_POOL
    
    health_status = {
        "status": "ok" if is_initialized else "initializing",
        "database": "connected" if DB_POOL is not None else "disconnected",
        "embedding_model": "loaded" if get_embed_model() is not None else "not_loaded",
        "timestamp": int(time.time())
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
    
    return health_status

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)