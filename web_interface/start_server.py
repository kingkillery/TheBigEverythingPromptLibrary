#!/usr/bin/env python3
"""
Startup script for The Big Everything Prompt Library Web Interface
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nTo install requirements, run:")
        print("pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    if not check_requirements():
        return
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    print("🚀 Starting The Big Everything Prompt Library Web Interface...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔍 API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()