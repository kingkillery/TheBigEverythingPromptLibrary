#!/usr/bin/env python3
"""
Test script for the web interface
Run this to validate the setup before starting the server
"""

import sys
import os
from pathlib import Path

# Add the backend directory to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        # Test gptparser import
        repo_root = Path(__file__).parent.parent
        sys.path.append(str(repo_root / ".scripts"))
        import gptparser
        print("✅ GPT parser imported successfully")
    except ImportError as e:
        print(f"❌ GPT parser import failed: {e}")
        return False
    
    # Test semantic search (optional)
    try:
        from semantic_search import create_semantic_search_engine
        print("✅ Semantic search available")
    except ImportError:
        print("ℹ️ Semantic search not available (optional)")
    
    return True

def test_repository_structure():
    """Test that the repository structure is as expected"""
    print("\n🏗️ Testing repository structure...")
    
    repo_root = Path(__file__).parent.parent
    
    required_dirs = [
        "CustomInstructions/ChatGPT",
        "Guides",
        "Articles", 
        "SystemPrompts",
        "Security",
        ".scripts"
    ]
    
    for dir_path in required_dirs:
        full_path = repo_root / dir_path
        if full_path.exists():
            print(f"✅ Found {dir_path}")
        else:
            print(f"❌ Missing {dir_path}")
            return False
    
    # Test for gptparser.py
    gptparser_path = repo_root / ".scripts" / "gptparser.py"
    if gptparser_path.exists():
        print("✅ Found gptparser.py")
    else:
        print("❌ Missing gptparser.py")
        return False
    
    return True

def test_gpt_parsing():
    """Test GPT file parsing"""
    print("\n📄 Testing GPT file parsing...")
    
    try:
        # Add scripts to path
        repo_root = Path(__file__).parent.parent
        sys.path.append(str(repo_root / ".scripts"))
        import gptparser
        
        # Count GPT files
        gpt_count = 0
        for ok, gpt in gptparser.enum_gpts():
            if ok:
                gpt_count += 1
                if gpt_count == 1:  # Test first file
                    print(f"✅ Successfully parsed sample GPT: {gpt.get('title', 'No title')[:50]}")
            if gpt_count >= 5:  # Test first 5 files
                break
        
        print(f"✅ Found {gpt_count} parseable GPT files")
        return gpt_count > 0
        
    except Exception as e:
        print(f"❌ GPT parsing failed: {e}")
        return False

def test_app_initialization():
    """Test that the app can be initialized"""
    print("\n🚀 Testing app initialization...")
    
    try:
        from app import app, index_manager
        print("✅ App imported successfully")
        
        if len(index_manager.index) > 0:
            print(f"✅ Index built with {len(index_manager.index)} items")
            
            # Test search
            sample_search = index_manager.search("coding", limit=5)
            print(f"✅ Sample search returned {len(sample_search.items)} results")
            
            return True
        else:
            print("❌ Index is empty")
            return False
            
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Testing The Big Everything Prompt Library Web Interface\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Repository Structure", test_repository_structure),
        ("GPT Parsing", test_gpt_parsing),
        ("App Initialization", test_app_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! You can start the server with:")
        print("   python start_server.py")
    else:
        print("❌ Some tests failed. Please check the setup before starting the server.")
        print("\nCommon fixes:")
        print("- Install requirements: pip install -r requirements.txt")
        print("- Ensure you're in the correct directory")
        print("- Check that the repository structure is intact")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)