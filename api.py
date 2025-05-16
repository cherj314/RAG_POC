import sys, os, uvicorn, time, json, asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import project modules
from rag.retriever import search_postgres, get_embed_model
from rag.generator import generate_response, get_available_models, build_prompt
from rag.config import init_db_pool, get_db_connection, release_connection

# Initialize FastAPI app with CORS
app = FastAPI(title="RAG API", description="API for the RAG-based question answering system")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Configuration
is_initialized = False
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "openai").lower()
AVAILABLE_MODEL_TYPES = os.getenv("AVAILABLE_MODEL_TYPES", "openai").lower().split(",")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
QUERY_SIMILARITY_THRESHOLD = float(os.getenv("QUERY_SIMILARITY_THRESHOLD", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Data models
class ProposalRequest(BaseModel):
    query: str
    similarity_threshold: Optional[float] = QUERY_SIMILARITY_THRESHOLD
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
    model: Optional[str] = None
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    show_retrieved: Optional[bool] = True
    similarity_threshold: Optional[float] = QUERY_SIMILARITY_THRESHOLD
    
class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None
    score: float

# Initialization
@app.on_event("startup")
async def startup_event():
    global is_initialized
    try:
        init_db_pool()
        get_embed_model()
        is_initialized = True
    except Exception as e:
        print(f"Error during startup: {str(e)}")

def ensure_initialized():
    global is_initialized
    if not is_initialized:
        try:
            init_db_pool()
            get_embed_model()
            is_initialized = True
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to initialize components")

# Helper functions
def format_retrieved_chunks(chunks):
    return [{"content": doc[:300] + "..." if len(doc) > 300 else doc,
             "source": metadata.get("file_name", "Unknown") if metadata else "Unknown",
             "score": score} for doc, metadata, score in chunks]

def parse_model_info(model_string):
    if not model_string:
        return DEFAULT_MODEL_TYPE, None
    
    if "/" in model_string:
        parts = model_string.split("/", 1)
        model_type = parts[0].lower()
        model_name = parts[1] if len(parts) > 1 else None
        return (model_type, model_name) if model_type in AVAILABLE_MODEL_TYPES else (DEFAULT_MODEL_TYPE, model_name)
    
    # If model type not explicitly specified, use default with provided model name
    return DEFAULT_MODEL_TYPE, model_string

def format_error_response(error_message):
    return {
        "id": f"chatcmpl-error-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{DEFAULT_MODEL_TYPE}/{get_default_model_name()}",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"I encountered an error processing your request: {error_message}. The system administrators have been notified."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def get_default_model_name():
    """Get the default model name based on the model type"""
    if DEFAULT_MODEL_TYPE == "openai":
        return DEFAULT_OPENAI_MODEL or "gpt-4o"
    elif DEFAULT_MODEL_TYPE == "ollama":
        return DEFAULT_OLLAMA_MODEL or "llama3:instruct"
    else:
        return "unknown"

# Routing endpoints
@app.post("/api/generate-proposal")
async def generate_proposal(request: ProposalRequest):
    try:
        ensure_initialized()
        chunks = search_postgres(request.query, k=request.max_chunks, similarity_threshold=request.similarity_threshold)
        
        if not chunks:
            return {"success": False, "message": "No relevant content found for the query."}
        
        retrieved_chunks = format_retrieved_chunks(chunks)
        
        if request.show_retrieved_only:
            return {"success": True, "retrieved_chunks": retrieved_chunks, "metadata": {"query": request.query}}
        
        start_time = time.time()
        response = generate_response(
            build_prompt(chunks, request.query),
            model_type=request.model_type or DEFAULT_MODEL_TYPE,
            model=request.model_name
        )
        
        return {
            "success": True,
            "proposal": response,
            "retrieved_chunks": retrieved_chunks,
            "metadata": {
                "generation_time": f"{time.time() - start_time:.2f}s",
                "query": request.query,
                "model_type": request.model_type or DEFAULT_MODEL_TYPE,
                "model_name": request.model_name or "default"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

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

async def process_chat_completion(request: ChatRequest):
    try:
        ensure_initialized()
        last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        
        if not last_user_message:
            return format_error_response("No user message found in the request")
        
        model_type, model_name = parse_model_info(request.model)
        
        # If model name is not provided, use the default for the model type
        if not model_name:
            model_name = DEFAULT_OPENAI_MODEL if model_type == "openai" else DEFAULT_OLLAMA_MODEL
        
        chunks = search_postgres(last_user_message, k=5, similarity_threshold=getattr(request, 'similarity_threshold', QUERY_SIMILARITY_THRESHOLD))
        retrieved_chunks = format_retrieved_chunks(chunks)
        show_retrieved = getattr(request, 'show_retrieved', True)
        
        if chunks:
            prompt = build_prompt(chunks, last_user_message)
            
            if show_retrieved:
                retrieved_content = "📚 **Retrieved Relevant Information:**\n\n"
                for i, chunk_info in enumerate(retrieved_chunks, 1):
                    retrieved_content += f"**[{i}] Source: {chunk_info['source']} (Score: {chunk_info['score']:.3f})**\n"
                    chunk_text = chunk_info['content']
                    if not chunk_text.endswith('\n'):
                        chunk_text += '\n'
                    retrieved_content += f"{chunk_text}\n"
                retrieved_content += "---\n\n**Generating response based on these sources...**\n\n"
                
                generated_response = generate_response(
                    prompt, max_tokens=request.max_tokens, model_type=model_type,
                    model=model_name, temperature=request.temperature, preserve_formatting=True
                )
                generated_text = retrieved_content + generated_response
            else:
                generated_text = generate_response(
                    prompt, max_tokens=request.max_tokens, model_type=model_type,
                    model=model_name, temperature=request.temperature, preserve_formatting=True
                )
        else:
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
        
        # Create model identifier
        model_identifier = f"{model_type}/{model_name}"
        
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_identifier,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(generated_text.split())
            }
        }
        
        return StreamingResponse(stream_response(generated_text, model_identifier)) if request.stream else response
    
    except Exception as e:
        print(f"Error in process_chat_completion: {str(e)}")
        import traceback
        traceback.print_exc()
        return format_error_response(f"I encountered an error processing your request: {str(e)}")

async def stream_response(content: str, model_identifier: str):
    paragraphs = content.split('\n\n')
    
    for i, paragraph in enumerate(paragraphs):
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_identifier,
            "choices": [{
                "index": 0,
                "delta": {"content": paragraph},
                "finish_reason": None
            }]
        }
        
        yield f"data: {json.dumps(data)}\n\n"
        
        if i < len(paragraphs) - 1:
            paragraph_break_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_identifier,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n\n"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(paragraph_break_data)}\n\n"
        
        await asyncio.sleep(0.05)
    
    end_data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_identifier,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(end_data)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/api/retrieve-chunks")
async def retrieve_chunks(request: ProposalRequest):
    try:
        ensure_initialized()
        chunks = search_postgres(request.query, k=request.max_chunks, similarity_threshold=request.similarity_threshold)
        
        return {
            "success": bool(chunks),
            "message": "No relevant content found for the query." if not chunks else None,
            "retrieved_chunks": format_retrieved_chunks(chunks) if chunks else None,
            "metadata": {"query": request.query} if chunks else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

@app.get("/models")
@app.get("/api/models")
async def list_models():
    try:
        available_models = get_available_models()
        model_data = []
        
        for model_type, models in available_models.items():
            model_data.extend([{
                "id": f"{model_type}/{model}", 
                "object": "model", 
                "created": int(time.time()),
                "owned_by": model_type, 
                "permission": [], 
                "root": model, 
                "parent": None
            } for model in models])
        
        return {"object": "list", "data": model_data}
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return {"object": "list", "data": []}

@app.get("/api/health")
async def health_check():
    from rag.config import DB_POOL
    
    health_status = {
        "status": "ok" if is_initialized else "initializing",
        "database": "connected" if DB_POOL is not None else "disconnected",
        "embedding_model": "loaded" if get_embed_model() is not None else "not_loaded",
        "timestamp": int(time.time())
    }
    
    if DB_POOL is not None:
        try:
            conn = get_db_connection()
            health_status["database_test"] = "success" if conn else "failed"
            if conn:
                release_connection(conn)
        except Exception as e:
            health_status["database_test"] = f"error: {str(e)}"
    
    return health_status

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)