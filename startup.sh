#!/bin/bash
# startup.sh - Detailed startup script with health checks
# Make sure to run: chmod +x ./startup.sh

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start it first."
  exit 1
fi

# Start the containers
echo "Starting RAGbot containers..."
docker compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check if API is running
echo "Checking if API is running..."
if curl -s http://localhost:8000/api/health | grep -q "status"; then
  echo "API is running!"
else
  echo "API is not responding. Check the logs with: docker compose logs ragbot-api"
fi

# Setup OpenWebUI
echo "Setting up OpenWebUI..."
echo "Web interface will be available at http://localhost:3000"
echo "Use the default token from your .env file to log in"
echo "Make sure to set the API base URL to http://localhost:8000 in the OpenWebUI settings"

# Complete
echo "RAGbot startup complete!"