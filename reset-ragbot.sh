#!/bin/bash
# reset-ragbot.sh - Reset the RAGbot system
# Make sure to run: chmod +x ./reset-ragbot.sh

echo "Stopping all containers..."
docker compose down

echo "Removing volumes..."
docker volume rm ragbot_pgdata ragbot_openwebui-data 2>/dev/null || true

echo "Pruning unused containers and networks..."
docker container prune -f
docker network prune -f

echo "Starting clean system..."
docker compose up -d postgres

echo "Waiting for PostgreSQL to initialize..."
sleep 10

docker compose up -d

echo "System has been reset and restarted!"