#!/bin/bash

# Default model
MODEL=${1:-tinyllama}

echo "Installing the $MODEL model into Ollama..."

# Wait for Ollama to be ready
echo "Waiting for Ollama service to be available..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "Ollama service is up!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "Timed out waiting for Ollama service"
    exit 1
  fi
  echo "Waiting..."
  sleep 2
done

# Pull the model
echo "Pulling $MODEL model..."
curl -X POST http://localhost:11434/api/pull -d "{\"name\":\"$MODEL\"}"

echo "Model installation initiated. This may take some time depending on your internet connection."
echo "You can check the status by running 'docker logs $(docker ps -qf "name=ragbot-ollama")'."
echo "Once complete, you'll be able to select the model in OpenWebUI."