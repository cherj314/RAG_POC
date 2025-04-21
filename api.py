import sys
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
import asyncio
import traceback

# Add parent directory to path to import project modules
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
    allow_origins=["*"],  # For production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global flag to track initialization status
is_initialized = False

# Initialize database pool on startup
@app.on_event("startup")
async def startup_event():
    global is_initialized
    try:
        print("🚀 Starting RAGbot API service...")
        # Initialize database pool
        init_db_pool()
        print("✅ Database connection pool initialized")
        
        # Pre-load embedding model
        model = get_embed_model()
        print(f"✅ Embedding model loaded: {model is not None}")
        
        is_initialized = True
        print("✅ RAGbot API initialization complete")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

# Define request and response models
class ProposalRequest(BaseModel):
    query: str
    similarity_threshold: Optional[float] = 0.5
    max_chunks: Optional[int] = 5

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    model: Optional[str] = "ragbot"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    
class ChatResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: List[Dict[str, Any]]

class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None
    score: float

# Ensure components are initialized
def ensure_initialized():
    """Ensure all required components are initialized"""
    global is_initialized
    
    if not is_initialized:
        print("⚠️ Components not initialized during startup, initializing now...")
        try:
            # Initialize database pool
            init_db_pool()
            print("✅ Database connection pool initialized")
            
            # Pre-load embedding model
            model = get_embed_model()
            print(f"✅ Embedding model loaded: {model is not None}")
            
            is_initialized = True
            print("✅ Manual initialization complete")
        except Exception as e:
            print(f"❌ Error during manual initialization: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Failed to initialize components")

# Routes
@app.post("/api/generate-proposal")
async def generate_proposal(request: ProposalRequest):
    try:
        # Ensure components are initialized
        ensure_initialized()
        
        # Step 1: Retrieve relevant chunks
        print(f"🔍 Searching for content relevant to: {request.query}")
        chunks = search_postgres(
            request.query,
            k=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        if not chunks:
            print("⚠️ No relevant content found for the query.")
            return {
                "success": False,
                "message": "No relevant content found for the query."
            }
        
        print(f"✅ Found {len(chunks)} relevant chunks")
        for i, (doc, metadata, score) in enumerate(chunks, 1):
            src = metadata.get("file_name", "Unknown") if metadata else "Unknown"
            print(f"  Chunk {i}: {src} (score: {score:.4f})")
        
        # Step 2: Build prompt
        prompt = build_prompt(chunks, request.query)
        print(f"✅ Built prompt with {len(chunks)} chunks")
        
        # Step 3: Generate response
        start_time = time.time()
        print("🧠 Generating response...")
        response = generate_response(prompt)
        generation_time = time.time() - start_time
        print(f"✅ Response generated in {generation_time:.2f}s")
        
        # Step 4: Format retrieved chunks for response
        retrieved_chunks = []
        for doc, metadata, score in chunks:
            source = metadata.get("file_name", "Unknown") if metadata else "Unknown"
            retrieved_chunks.append({
                "content": doc[:300] + "..." if len(doc) > 300 else doc,
                "source": source,
                "score": score
            })
        
        return {
            "success": True,
            "proposal": response,
            "metadata": {
                "chunks": retrieved_chunks,
                "generation_time": f"{generation_time:.2f}s",
                "query": request.query
            }
        }
    except Exception as e:
        print(f"❌ Error generating proposal: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

@app.post("/chat/completions")
async def chat_completions_no_prefix(request: ChatRequest, raw_request: Request):
    """
    OpenAI-compatible chat completions endpoint (without /api prefix)
    """
    return await chat_completions(request, raw_request)

@app.post("/api/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    """
    OpenAI-compatible chat completions endpoint for OpenWebUI integration
    """
    try:
        # Ensure components are initialized
        ensure_initialized()
        
        # Extract all messages for context
        print("\n==== CHAT REQUEST RECEIVED ====")
        print(f"Model: {request.model}, Temperature: {request.temperature}, Max tokens: {request.max_tokens}")
        
        # Debug: Print all messages
        print(f"Messages in request: {len(request.messages)}")
        for i, msg in enumerate(request.messages):
            preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  Message {i+1}: {msg.role} - {preview}")
        
        # Extract the last user message as our query
        last_user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        if not last_user_message:
            error_msg = "No user message found in the request"
            print(f"❌ Error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        print(f"🔍 Query for RAG: \"{last_user_message}\"")
        
        # Process the query using our RAG pipeline
        print("🔍 Searching for relevant content...")
        search_start = time.time()
        chunks = search_postgres(
            last_user_message,
            k=5,
            similarity_threshold=0.5
        )
        search_time = time.time() - search_start
        
        # Log retrieved chunks
        if chunks:
            print(f"✅ Found {len(chunks)} relevant chunks in {search_time:.2f}s")
            for i, (doc, metadata, score) in enumerate(chunks, 1):
                src = metadata.get("file_name", "Unknown") if metadata else "Unknown"
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"  Chunk {i}: {src} (score: {score:.4f})")
                print(f"    Preview: {preview}")
        else:
            print("⚠️ No relevant chunks found, falling back to base knowledge")
        
        # Build the prompt and generate a response
        if chunks:
            print("🧠 Building prompt with retrieved context...")
            prompt = build_prompt(chunks, last_user_message)
            
            print("🧠 Generating response with context from retrieved chunks...")
            generated_text = generate_response(prompt, max_tokens=request.max_tokens)
        else:
            # Don't generate a response, instead return a message about no relevant content
            print("⚠️ No relevant chunks found, informing user")
            generated_text = (
                "I'm sorry, but I couldn't find any relevant information in my knowledge base "
                "about your query. Could you please rephrase your question or ask about something "
                "related to software development proposals, which is my area of expertise?"
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
        
        print(f"✅ Response generated successfully ({len(generated_text.split())} tokens)")
        response_preview = generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
        print(f"  Preview: {response_preview}")
        print("==== REQUEST COMPLETE ====\n")
        
        # If streaming is requested
        if request.stream:
            print("Streaming response requested")
            return StreamingResponse(stream_response(generated_text))
        
        return response
    
    except Exception as e:
        # Detailed error logging
        error_msg = f"Error in chat completion: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Return a proper error response that OpenWebUI can understand
        response = {
            "id": f"chatcmpl-error-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or "ragbot",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"I encountered an error processing your request: {str(e)}. The RAGbot system administrators have been notified."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": 0,
                "total_tokens": sum(len(m.content.split()) for m in request.messages)
            }
        }
        
        return response

async def stream_response(content: str):
    """Generate streaming response chunks for OpenAI-compatible streaming"""
    # Split content into reasonable chunks
    chunks = []
    words = content.split()
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= 10:  # Send ~10 words at a time
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Stream each chunk
    for i, chunk in enumerate(chunks):
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "ragbot",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk + " "
                    },
                    "finish_reason": None if i < len(chunks) - 1 else "stop"
                }
            ]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)  # Small delay to simulate typing
    
    # End the stream
    yield "data: [DONE]\n\n"

from fastapi.responses import StreamingResponse

@app.get("/models")
async def list_models_no_prefix():
    """OpenAI-compatible models endpoint (without /api prefix)"""
    return await list_models()

@app.get("/api/models")
async def list_models():
    """OpenAI-compatible models endpoint"""
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
    # Check if the database pool is initialized
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