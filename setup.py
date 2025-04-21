#!/usr/bin/env python3
"""
RAGbot Setup Script - Prepares environment for running RAGbot
"""
import os
import subprocess
import platform
import shutil
import sys
from pathlib import Path

def print_step(message):
    """Print a formatted step message"""
    print(f"\n{'='*80}\n{message}\n{'='*80}")

def check_prerequisites():
    """Check if required tools are installed"""
    print_step("Checking prerequisites...")
    
    # Check Python version
    python_version = platform.python_version_tuple()
    if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {platform.python_version()} detected")
    
    # Check Docker and Docker Compose
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Docker not found. Please install Docker: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    try:
        # Try to use docker compose command first (newer Docker versions)
        try:
            subprocess.run(["docker", "compose", "version"], check=True, capture_output=True)
            print("✅ Docker Compose (new CLI) installed")
        except subprocess.SubprocessError:
            # Fall back to docker-compose command (older versions)
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
            print("✅ Docker Compose (classic) installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Docker Compose not found. Please install Docker Compose: https://docs.docker.com/compose/install/")
        sys.exit(1)
    
    print("All prerequisites are satisfied!")

def setup_environment():
    """Set up the environment variables"""
    print_step("Setting up environment variables...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print(f"❌ {env_example} not found. Please make sure you're in the RAGbot directory.")
        sys.exit(1)
    
    if env_file.exists():
        overwrite = input(".env file already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Keeping existing .env file.")
            return
    
    # Copy .env.example to .env
    shutil.copy(env_example, env_file)
    print(f"✅ Created .env file from {env_example}")
    
    # Prompt for OpenAI API key
    api_key = input("Enter your OpenAI API key (leave blank to configure later): ").strip()
    if api_key:
        # Read the content of the .env file
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace the placeholder with the actual API key
        updated_content = content.replace("OPENAI_API_KEY=your_openai_api_key_here", f"OPENAI_API_KEY={api_key}")
        
        # Write the updated content back to the .env file
        with open(env_file, 'w') as f:
            f.write(updated_content)
        
        print("✅ Updated OpenAI API key in .env file")

def setup_virtual_environment():
    """Set up Python virtual environment"""
    print_step("Setting up Python virtual environment...")
    
    venv_dir = "venv"
    
    # Check if venv already exists
    if os.path.exists(venv_dir):
        overwrite = input(f"{venv_dir}/ already exists. Recreate? (y/n): ").lower()
        if overwrite == 'y':
            # Remove existing venv
            if platform.system() == "Windows":
                os.system(f"rmdir /s /q {venv_dir}")
            else:
                os.system(f"rm -rf {venv_dir}")
        else:
            print(f"Keeping existing {venv_dir}/ directory.")
            return
    
    # Create virtual environment
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print(f"✅ Created virtual environment in {venv_dir}/")
    except subprocess.SubprocessError:
        print(f"❌ Failed to create virtual environment in {venv_dir}/")
        sys.exit(1)
    
    # Determine pip path based on platform
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    # Install dependencies
    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Installed dependencies")
    except subprocess.SubprocessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def create_documents_directory():
    """Create Documents directory if it doesn't exist"""
    print_step("Setting up Documents directory...")
    
    docs_dir = "Documents"
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"✅ Created {docs_dir}/ directory")
        
        # Copy sample documents if they exist in the repo
        for sample_file in ["project_capabilities.txt", "project_proposals.txt"]:
            if os.path.exists(sample_file):
                shutil.copy(sample_file, os.path.join(docs_dir, sample_file))
                print(f"✅ Copied {sample_file} to {docs_dir}/")
    else:
        print(f"✅ {docs_dir}/ directory already exists")

def print_final_instructions():
    """Print final setup instructions"""
    print_step("🎉 Setup Complete!")
    
    print("""
RAGbot environment has been set up successfully!

To start the system:

1. Start the database and services:
   docker compose up -d

2. Activate the virtual environment:
   - On Windows: venv\\Scripts\\activate
   - On Linux/Mac: source venv/bin/activate

3. Ingest your documents (only needed once or when documents change):
   python ingest.py

4. Run the application in one of these ways:
   - Interactive mode: python main.py
   - API server: python -m api

5. Access the web interface at:
   http://localhost:3000

For more information, refer to the README.md file.
""")

def main():
    """Main setup function"""
    print("""
🧠 RAGbot Setup
--------------
This script will set up the environment for running RAGbot.
""")
    
    try:
        check_prerequisites()
        setup_environment()
        setup_virtual_environment()
        create_documents_directory()
        print_final_instructions()
    except KeyboardInterrupt:
        print("\n\nSetup aborted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()