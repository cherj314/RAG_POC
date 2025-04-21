.PHONY: setup start stop restart status logs clean ingest shell reset reset-data help

# Default target
help:
	@echo "RAGbot Make Commands"
	@echo "===================="
	@echo "setup    - Run setup script to configure environment"
	@echo "start    - Start all containers"
	@echo "stop     - Stop all containers"
	@echo "restart  - Restart all containers"
	@echo "status   - Check container status"
	@echo "logs     - View API logs"
	@echo "logs-ingest - View ingest logs"
	@echo "ingest   - Run document ingestion manually"
	@echo "shell    - Open a shell in the API container"
	@echo "reset    - Stop containers, remove volumes, and restart (CAUTION: data loss)"
	@echo "reset-data - Reset only the database content without rebuilding containers"
	@echo "clean    - Remove all containers and volumes (CAUTION: data loss)"
	@echo "help     - Show this help message"

# Setup environment
setup:
	python setup.py

# Start all containers
start:
	docker compose up -d
	@echo "Services starting..."
	@echo "Web interface will be available at http://localhost:3000"

# Start with verbose output
start-verbose:
	@echo "Starting RAGbot with verbose output..."
	@chmod +x ./startup.sh
	./startup.sh

# Stop all containers
stop:
	docker compose down

# Restart all containers
restart:
	docker compose restart

# Check container status
status:
	docker compose ps

# View API logs
logs:
	docker compose logs -f ragbot-api

# View ingestion logs
logs-ingest:
	docker compose logs ragbot-ingest

# Run document ingestion (only if needed manually)
ingest:
	docker compose up ragbot-ingest

# Open a shell in the API container
shell:
	docker compose exec ragbot-api /bin/bash

# Reset the system (stop, remove volumes, restart)
reset:
	@echo "Stopping all containers..."
	docker compose down
	@echo "Removing volumes..."
	docker volume rm rag_poc_pgdata rag_poc_openwebui-data || true
	@echo "Pruning unused containers and networks..."
	docker container prune -f
	docker network prune -f
	@echo "Starting clean system..."
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to initialize..."
	powershell -Command "Start-Sleep -Seconds 10"
	docker compose up -d
	@echo "System has been reset and restarted!"

# Reset only the database content without rebuilding
reset-data:
	docker compose stop postgres
	docker compose rm -f postgres
	docker volume rm ragbot_pgdata || true
	docker compose up -d
	@echo "Database has been reset. System is restarting with fresh data..."

# Clean everything (CAUTION: data loss)
clean:
	@echo "WARNING: This will remove all containers, volumes, and data!"
	@read -p "Are you sure? (y/n) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		docker compose down -v; \
		echo "Cleaned all containers and volumes"; \
	else \
		echo "Clean operation cancelled"; \
	fi