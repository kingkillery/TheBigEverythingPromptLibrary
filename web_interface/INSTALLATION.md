# Installation Guide

## Quick Start

1. **Install Python Dependencies**
   ```bash
   cd web_interface
   pip install -r requirements.txt
   ```

2. **Test the Setup**
   ```bash
   python test_interface.py
   ```

3. **Start the Server**
   ```bash
   python start_server.py
   ```

4. **Access the Interface**
   - Open your browser to: http://localhost:8000
   - API docs available at: http://localhost:8000/docs

## Enhanced Setup (Optional Semantic Search)

For advanced semantic search capabilities:

1. **Install Additional Dependencies**
   ```bash
   pip install sentence-transformers numpy
   ```

2. **The system will automatically enable semantic search features**

## System Requirements

- Python 3.8 or higher
- 2GB RAM (4GB recommended for semantic search)
- 1GB disk space for embeddings cache (if using semantic search)

## Troubleshooting

### "No module named 'fastapi'" Error
```bash
pip install -r requirements.txt
```

### "No module named 'gptparser'" Error
Ensure you're running from the `web_interface` directory.

### Empty Search Results
Click the "Refresh Index" button to rebuild the search index.

### Port 8000 Already in Use
The server will show an error. Kill any existing processes on port 8000:
```bash
# On Windows
netstat -ano | findstr :8000
taskkill /PID [PID_NUMBER] /F

# On Linux/Mac
lsof -ti:8000 | xargs kill -9
```

## Performance Notes

- Initial indexing: 2-3 seconds
- Search response time: <100ms
- Memory usage: ~50MB (base) + ~200MB (with semantic search)
- Index refresh: ~2 seconds for 1,800+ items