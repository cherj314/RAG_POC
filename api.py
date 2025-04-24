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
    show_retrieved_only: Optional[bool] = False  # New parameter to show only retrieved chunks

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    model: Optional[str] = "ragbot"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    show_retrieved: Optional[bool] = True  # New parameter to control retrieval visibility
    
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
        
        # Step 1: Retrieve relevant chunks
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
        
        # If user only wants to see retrieved chunks, return them without generating a proposal
        if request.show_retrieved_only:
            return {
                "success": True,
                "retrieved_chunks": retrieved_chunks,
                "metadata": {
                    "query": request.query
                }
            }
        
        # Step 2: Build prompt
        prompt = build_prompt(chunks, request.query)
        
        # Step 3: Generate response
        start_time = time.time()
        response = generate_response(prompt)
        generation_time = time.time() - start_time
        
        return {
            "success": True,
            "proposal": response,
            "retrieved_chunks": retrieved_chunks,
            "metadata": {
                "generation_time": f"{generation_time:.2f}s",
                "query": request.query
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

@app.post("/chat/completions")
async def chat_completions_no_prefix(request: ChatRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint (without /api prefix)"""
    return await chat_completions(request, raw_request)

@app.post("/api/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint for OpenWebUI integration"""
    try:
        ensure_initialized()
        
        # Extract the last user message as our query
        last_user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found in the request")
        
        # Process the query using our RAG pipeline
        chunks = search_postgres(
            last_user_message,
            k=5,
            similarity_threshold=0.5
        )
        
        # Format retrieved chunks
        retrieved_chunks = format_retrieved_chunks(chunks)
        
        # Check if show_retrieved is set to True or not provided (default is True)
        show_retrieved = request.show_retrieved if hasattr(request, 'show_retrieved') else True
        
        # Prepare response content
        if chunks:
            # Build the prompt with retrieved chunks
            prompt = build_prompt(chunks, last_user_message)
            
            # If show_retrieved is True, first show the retrieved chunks
            if show_retrieved:
                # Prepare a message that displays the retrieved chunks
                retrieved_content = "📚 **Retrieved Relevant Information:**\n\n"
                for i, chunk_info in enumerate(retrieved_chunks, 1):
                    retrieved_content += f"**[{i}] Source: {chunk_info['source']} (Score: {chunk_info['score']:.3f})**\n"
                    # Format the chunk content with proper paragraph breaks
                    chunk_text = chunk_info['content']
                    # Ensure content ends with a newline
                    if not chunk_text.endswith('\n'):
                        chunk_text += '\n'
                    retrieved_content += f"{chunk_text}\n"
                retrieved_content += "---\n\n**Generating proposal based on these sources...**\n\n"
                
                # Add the chunks first, then generate the response with formatting preserved
                generated_response = generate_response(prompt, max_tokens=request.max_tokens, preserve_formatting=True)
                
                # Ensure there's a proper paragraph break between retrieved content and generated text
                if not generated_response.startswith('\n'):
                    generated_text = retrieved_content + generated_response
                else:
                    generated_text = retrieved_content + generated_response
            else:
                # Just generate the response without showing retrieved chunks
                generated_text = generate_response(prompt, max_tokens=request.max_tokens, preserve_formatting=True)
        else:
            # No relevant chunks found
            if show_retrieved:
                generated_text = (
                    "❌ **No relevant information found in my knowledge base.**\n\n"
                    "I'm sorry, but I couldn't find any relevant information in my knowledge base "
                    "about your query. Could you please rephrase your question or ask about something "
                    "related to software development proposals, which is my area of expertise?"
                )
            else:
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
        
        # If streaming is requested
        if request.stream:
            return StreamingResponse(stream_response(generated_text))
        
        return response
    
    except Exception as e:
        # Return a proper error response
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

# New endpoint to get only retrieved chunks without generating a response
@app.post("/api/retrieve-chunks")
async def retrieve_chunks(request: ProposalRequest):
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
        
        return {
            "success": True,
            "retrieved_chunks": retrieved_chunks,
            "metadata": {
                "query": request.query
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

async def stream_response(content: str):
    """Generate streaming response chunks for OpenAI-compatible streaming with proper paragraph formatting"""
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
                    "delta": {
                        "content": retrieved_content
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.1)  # Slightly longer pause after retrieved content
        
        # Update content to be just the proposal part
        content = proposal_content
    
    # Split the content by paragraphs first
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
                                "delta": {
                                    "content": chunk_text + " "
                                },
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
                            "delta": {
                                "content": chunk_text
                            },
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
                        "delta": {
                            "content": paragraph
                        },
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
                        "delta": {
                            "content": "\n\n"
                        },
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