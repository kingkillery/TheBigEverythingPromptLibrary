# Quick Start Guide

## Method 1: Direct Command (Recommended)

**Windows PowerShell:**
```powershell
cd web_interface\backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Windows Command Prompt:**
```cmd
cd web_interface\backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Linux/Mac:**
```bash
cd web_interface/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Method 2: Using Scripts

**Windows:**
```cmd
web_interface\run_server.bat
```

**Linux/Mac:**
```bash
./web_interface/run_server.sh
```

**Cross-platform Python:**
```bash
python web_interface/start_server.py
```

## Method 3: One-Liner from Repository Root

**Windows:**
```powershell
cd web_interface\backend && python -m uvicorn app:app --reload && cd ..\..
```

**Linux/Mac:**
```bash
cd web_interface/backend && python -m uvicorn app:app --reload && cd ../..
```

## Access the Interface

Once started, open your browser to:
- **Main Interface:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## Troubleshooting

### "No module named 'app'" Error
Make sure you're running from the `backend` directory:
```bash
cd web_interface/backend
python -m uvicorn app:app --reload
```

### "No module named 'fastapi'" Error
Install requirements first:
```bash
pip install -r web_interface/requirements.txt
```

### "Permission denied" Error (Linux/Mac)
Make the script executable:
```bash
chmod +x web_interface/run_server.sh
```

## Success Output

You should see:
```
Building search index...
Processing GPT files...
Processing Guides...
Processing Articles...
Processing SystemPrompts...
Processing Security...
Index built with 1800+ items
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Adding LLM Features (Optional)

1. Get free API key from https://openrouter.ai/keys
2. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```
3. Restart the server to see LLM enhancement buttons