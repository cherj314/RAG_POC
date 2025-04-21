#!/usr/bin/env python3
"""
Test script for RAGbot API endpoints.
Run this script to ensure that the API is functioning correctly.
"""
import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the API health endpoint."""
    url = f"{API_BASE_URL}/api/health"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        if result.get("status") == "ok":
            print("✅ Health check endpoint is working")
            return True
        else:
            print("❌ Health check endpoint returned unexpected response:", result)
            return False
    except Exception as e:
        print(f"❌ Health check endpoint failed: {e}")
        return False

def test_models():
    """Test the models endpoint."""
    url = f"{API_BASE_URL}/api/models"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        if result.get("object") == "list" and len(result.get("data", [])) > 0:
            print("✅ Models endpoint is working")
            print(f"   Available models: {', '.join([model['id'] for model in result['data']])}")
            return True
        else:
            print("❌ Models endpoint returned unexpected response:", result)
            return False
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False

def test_chat_completions():
    """Test the chat completions endpoint."""
    url = f"{API_BASE_URL}/api/chat/completions"
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Create a short proposal for a mobile inventory application."
            }
        ],
        "model": "ragbot",
        "max_tokens": 500
    }
    
    print("📤 Sending request to chat completions endpoint...")
    print("   This may take a moment as it needs to search the vector database and generate a response...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time
        
        if result.get("choices") and len(result["choices"]) > 0:
            print(f"✅ Chat completions endpoint is working (took {elapsed:.2f}s)")
            content = result["choices"][0]["message"]["content"]
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"   Response preview: {preview}")
            return True
        else:
            print("❌ Chat completions endpoint returned unexpected response:", result)
            return False
    except Exception as e:
        print(f"❌ Chat completions endpoint failed: {e}")
        return False

def test_proposal_endpoint():
    """Test the generate-proposal endpoint."""
    url = f"{API_BASE_URL}/api/generate-proposal"
    data = {
        "query": "Create a proposal for a web-based inventory system with barcode scanning",
        "similarity_threshold": 0.5,
        "max_chunks": 5
    }
    
    print("📤 Sending request to generate-proposal endpoint...")
    print("   This may take a moment as it needs to search the vector database and generate a response...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time
        
        if result.get("success") == True and "proposal" in result:
            print(f"✅ Generate-proposal endpoint is working (took {elapsed:.2f}s)")
            preview = result["proposal"][:150] + "..." if len(result["proposal"]) > 150 else result["proposal"]
            print(f"   Response preview: {preview}")
            
            # Print retrieved chunks
            print("\nRetrieved chunks:")
            for i, chunk in enumerate(result.get("metadata", {}).get("chunks", []), 1):
                print(f"  {i}. {chunk['source']} (score: {chunk['score']:.4f})")
                
            return True
        else:
            print("❌ Generate-proposal endpoint returned unexpected response:", result)
            return False
    except Exception as e:
        print(f"❌ Generate-proposal endpoint failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing RAGbot API endpoints...\n")
    
    # Allow specifying a different API base URL
    global API_BASE_URL
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
        print(f"Using custom API base URL: {API_BASE_URL}")
    
    health_ok = test_health()
    print()
    
    models_ok = test_models()
    print()
    
    chat_ok = test_chat_completions()
    print()
    
    proposal_ok = test_proposal_endpoint()
    print()
    
    # Summary
    print("📋 Test summary:")
    print(f"   Health endpoint: {'✅' if health_ok else '❌'}")
    print(f"   Models endpoint: {'✅' if models_ok else '❌'}")
    print(f"   Chat completions endpoint: {'✅' if chat_ok else '❌'}")
    print(f"   Generate-proposal endpoint: {'✅' if proposal_ok else '❌'}")
    
    all_passing = all([health_ok, models_ok, chat_ok, proposal_ok])
    if all_passing:
        print("\n🎉 All tests passed! Your RAGbot API is working correctly.")
        print("   Now you can access OpenWebUI at: http://localhost:3000")
    else:
        print("\n⚠️ Some tests failed. Please check the logs for more information.")

if __name__ == "__main__":
    main()