#!/usr/bin/env python
"""RAGbot Setup Script"""

import os
import subprocess
from pathlib import Path
import platform
import re

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"

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
    if IS_WINDOWS:
        # Use plain print for Windows to avoid PowerShell issues
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
        # Use ANSI colors on Unix-like systems
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
        "DB_HOST": "postgres",
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
    
    if run_command("docker --version"):
        print_color("Docker is installed", "green")
    else:
        print_color("Docker is not installed or not in PATH", "red")
        print_color("Please install Docker Desktop from https://www.docker.com/products/docker-desktop", "yellow")
        return False
    
    # Check docker-compose
    if run_command("docker compose version"):
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
    if IS_WINDOWS:
        print_color("run-ragbot.bat", "yellow")
    else:
        print_color("docker compose up -d", "yellow")
    print_color("\nTo reset the system, run:", "blue")
    if IS_WINDOWS:
        print_color("reset-ragbot.bat", "yellow")
    else:
        print_color("docker compose down && docker volume rm ragbot_pgdata", "yellow")
    print_color("\nThe web interface will be available at:", "blue")
    print_color("http://localhost:3000", "yellow")

if __name__ == "__main__":
    main()