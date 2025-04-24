#!/bin/bash
# run-ragbot.sh - Start the RAGbot system
# Make sure to run: chmod +x ./run-ragbot.sh

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start it first."
  exit 1
fi

# Start the containers
docker compose up -d

echo "Services starting..."
sleep 5

echo "Web interface will be available at http://localhost:3000"
echo "Use the default token from your .env file to log in"
echo "Make sure to set the API base URL to http://localhost:8000 in the OpenWebUI settings"