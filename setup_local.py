#!/usr/bin/env python3
"""
Setup script for Delhi High Court Virtual Judge application
This script helps install all dependencies needed to run the application locally.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies from requirements_local.txt"""
    print("Installing dependencies from requirements_local.txt...")
    
    try:
        # Check if pip is available
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_local.txt"
        ])
        
        print("Successfully installed dependencies!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def setup_data_directories():
    """Create necessary data directories if they don't exist"""
    print("Setting up data directories...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/embeddings"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Delhi High Court Virtual Judge - Local Setup")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Setup data directories
    setup_data_directories()
    
    print("\nSetup complete! You can now run the application with:")
    print("streamlit run app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()