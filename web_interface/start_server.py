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
    
    # Get the absolute path to the backend directory
    backend_dir = Path(__file__).parent / "backend"
    backend_abs_path = str(backend_dir.resolve())
    
    print("🚀 Starting The Big Everything Prompt Library Web Interface...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔍 API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        import uvicorn
        import sys
        
        # Add backend directory to Python path
        if backend_abs_path not in sys.path:
            sys.path.insert(0, backend_abs_path)
        
        # Change to backend directory for relative imports
        original_cwd = os.getcwd()
        os.chdir(backend_dir)
        
        # Start the server
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, 
                   reload_dirs=[backend_abs_path])
                   
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Try running from the backend directory directly:")
        print(f"   cd {backend_dir}")
        print("   python -m uvicorn app:app --reload")
    finally:
        # Restore original directory
        try:
            os.chdir(original_cwd)
        except:
            pass

if __name__ == "__main__":
    start_server()