#!/usr/bin/env python
"""RAGbot Setup Script"""

import os
import subprocess
from pathlib import Path
import platform
import re

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"
# Check if running on WSL
IS_WSL = "microsoft-standard" in platform.uname().release.lower() if platform.system() == "Linux" else False

def print_color(text, color):
    """Print colored text to the console."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "bold": "\033[1m",
        "end": "\033[0m"
    }
    if IS_WINDOWS and not IS_WSL:
        # Use plain print for Windows CMD to avoid PowerShell issues
        if color == "green":
            print(f"✓ {text}")
        elif color == "yellow":
            print(f"! {text}")
        elif color == "red":
            print(f"✗ {text}")
        elif color == "blue":
            print(f"→ {text}")
        else:
            print(text)
    else:
        # Use ANSI colors on Unix-like systems or Windows Terminal
        print(f"{colors.get(color, '')}{text}{colors['end']}")

def run_command(command, shell=False):
    """Run a shell command and return the output."""
    try:
        if isinstance(command, str) and not shell:
            command = command.split()
        result = subprocess.run(command, capture_output=True, text=True, shell=shell)
        if result.returncode != 0:
            print_color(f"Command failed: {command}", "red")
            print_color(f"Error: {result.stderr}", "red")
            return False
        return result.stdout
    except Exception as e:
        print_color(f"Error executing command: {e}", "red")
        return False

def create_env_file():
    """Create or update the .env file with required variables."""
    print_color("\nSetting up environment variables...", "blue")
    
    # Default values
    env_vars = {
        "POSTGRES_USER": "myuser",
        "POSTGRES_PASSWORD": "mypassword",
        "POSTGRES_DB": "vectordb",
        "DB_HOST": "postgres",  # Use 'postgres' for Docker networking
        "DB_PORT": "5432",
        "DB_NAME": "vectordb",
        "DB_USER": "myuser",
        "DB_PASSWORD": "mypassword",
        "API_PORT": "8000",
        "WEBUI_PORT": "3000",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "COLLECTION_NAME": "document_chunks",
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "50",
        "OPENAI_API_KEY": "",
        "WEBUI_AUTH_TOKEN": "default_token"
    }
    
    # Check if .env file exists
    env_file_exists = os.path.exists(".env")
    if env_file_exists:
        with open(".env", "r") as f:
            existing_content = f.read()
            
        # Extract existing values
        for key in env_vars.keys():
            match = re.search(f"{key}=(.*?)($|\n)", existing_content)
            if match:
                env_vars[key] = match.group(1)
    
    # Ask for OpenAI API key if not set
    if not env_vars["OPENAI_API_KEY"]:
        print_color("OpenAI API key is required for this application.", "yellow")
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            env_vars["OPENAI_API_KEY"] = api_key
        else:
            print_color("No API key provided. You will need to set this later.", "yellow")
    
    # Write to .env file
    with open(".env", "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    if env_file_exists:
        print_color("Updated existing .env file", "green")
    else:
        print_color("Created new .env file", "green")

def setup_documents_directory():
    """Create the Documents directory if it doesn't exist."""
    docs_dir = Path("Documents")
    if not docs_dir.exists():
        docs_dir.mkdir()
        print_color("Created Documents directory", "green")
        
        # Create a sample document for testing
        sample_doc = docs_dir / "sample.txt"
        with open(sample_doc, "w") as f:
            f.write("This is a sample document for testing RAGbot's retrieval capabilities.\n\n")
            f.write("RAGbot uses retrieval-augmented generation to provide accurate and contextually relevant responses based on your documents.\n\n")
    else:
        print_color("Documents directory already exists", "green")

def setup_docker_files():
    """Check and setup Docker files."""
    print_color("\nChecking Docker configuration...", "blue")
    
    # Check docker
    if run_command("docker --version"):
        print_color("Docker is installed", "green")
    else:
        print_color("Docker is not installed or not in PATH", "red")
        if IS_WINDOWS and not IS_WSL:
            print_color("Please install Docker from https://docs.docker.com/engine/install/", "yellow")
        else:
            print_color("Please install Docker Engine with: sudo apt-get install docker-ce docker-ce-cli containerd.io", "yellow")
        return False
    
    # Check docker-compose
    if run_command("docker compose version") or run_command("docker-compose version"):
        print_color("Docker Compose is installed", "green")
    else:
        print_color("Docker Compose is not installed or not in PATH", "red")
        print_color("Docker Compose is required for this application", "yellow")
        return False
    
    # Check if required Docker files exist
    required_files = ["docker-compose.yml", "Dockerfile", "Dockerfile-api"]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print_color(f"Missing Docker files: {', '.join(missing_files)}", "red")
        print_color("Please make sure all Docker files are in the project directory", "yellow")
        return False
    
    print_color("All Docker files are present", "green")
    
    # For Linux/WSL, create shell scripts
    if not IS_WINDOWS or IS_WSL:
        # Create the shell scripts if they don't exist
        if not os.path.exists("check-docker.sh"):
            with open("check-docker.sh", "w") as f:
                f.write("""#!/bin/bash
# Check if Docker is running
if docker info > /dev/null 2>&1; then
  echo "Docker is running"
  exit 0
else
  echo "Docker is not running. Please start it first."
  exit 1
fi
""")
            os.chmod("check-docker.sh", 0o755)
            print_color("Created check-docker.sh script", "green")
            
        if not os.path.exists("run-ragbot.sh"):
            with open("run-ragbot.sh", "w") as f:
                f.write("""#!/bin/bash
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
""")
            os.chmod("run-ragbot.sh", 0o755)
            print_color("Created run-ragbot.sh script", "green")
            
        if not os.path.exists("reset-ragbot.sh"):
            with open("reset-ragbot.sh", "w") as f:
                f.write("""#!/bin/bash
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
""")
            os.chmod("reset-ragbot.sh", 0o755)
            print_color("Created reset-ragbot.sh script", "green")
            
        if not os.path.exists("startup.sh"):
            with open("startup.sh", "w") as f:
                f.write("""#!/bin/bash
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
""")
            os.chmod("startup.sh", 0o755)
            print_color("Created startup.sh script", "green")
    
    return True

def main():
    """Main setup function."""
    print("\n===========================")
    print("  RAGbot Setup Wizard")
    print("===========================\n")
    
    # Setup environment variables
    create_env_file()
    
    # Setup Documents directory
    print_color("\nSetting up Documents directory...", "blue")
    setup_documents_directory()
    
    # Setup Docker files
    if not setup_docker_files():
        print_color("\nSetup cannot continue due to missing Docker components.", "red")
        return
    
    print_color("\nSetup completed successfully!", "green")
    print_color("\nTo start the system, run:", "blue")
    if IS_WINDOWS and not IS_WSL:
        print_color("docker compose up -d", "yellow")
    else:
        print_color("./run-ragbot.sh", "yellow")
    print_color("\nTo reset the system, run:", "blue")
    if IS_WINDOWS and not IS_WSL:
        print_color("docker compose down && docker volume rm ragbot_pgdata ragbot_openwebui-data", "yellow")
    else:
        print_color("./reset-ragbot.sh", "yellow")
    print_color("\nThe web interface will be available at:", "blue")
    print_color("http://localhost:3000", "yellow")

if __name__ == "__main__":
    main()