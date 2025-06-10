# The Big Everything Prompt Library - Web Interface

A modern web interface for searching and exploring the comprehensive prompt library containing 1,800+ AI prompts, custom instructions, guides, and resources.

## Features

🔍 **Powerful Search**: Full-text search across all prompts and content
📂 **Smart Filtering**: Filter by category, tags, and content type  
🏷️ **Tag System**: Automatic tag extraction and filtering
📱 **Responsive Design**: Works on desktop, tablet, and mobile
⚡ **Real-time Updates**: Auto-detects new files when added to repository
🎯 **Fuzzy Matching**: Find content even with partial or approximate searches

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation & Setup

1. **Install Dependencies**
   ```bash
   cd web_interface
   pip install -r requirements.txt
   ```

2. **Start the Server**
   ```bash
   python start_server.py
   ```

3. **Access the Interface**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Content Coverage

The web interface automatically indexes and makes searchable:

- **1,670+ ChatGPT Custom Instructions** (`CustomInstructions/ChatGPT/`)
- **Comprehensive Guides** (`Guides/`) - Prompt engineering, tool usage
- **Research Articles** (`Articles/`) - Educational content and analysis  
- **System Prompts** (`SystemPrompts/`) - Official prompts from AI platforms
- **Security Resources** (`Security/`) - Defense and best practices
- **Advanced Techniques** (`Jailbreak/`) - Educational research content

## API Endpoints

### Search
```
GET /api/search?query=coding&category=CustomInstructions&tags=python
```

### Categories
```
GET /api/categories
```

### Tags  
```
GET /api/tags
```

### Refresh Index
```
POST /api/refresh
```

### Item Details
```
GET /api/item/{item_id}
```

## Architecture

### Backend (FastAPI)
- **File Parsing**: Leverages existing `gptparser.py` for ChatGPT files
- **Indexing System**: Automatically scans all repository directories
- **Search Engine**: Text-based search with scoring and ranking
- **Category Detection**: Smart categorization based on file location and content
- **Tag Extraction**: Automatic tag generation from content analysis

### Frontend (Vanilla JavaScript)
- **Single Page App**: No framework dependencies
- **Real-time Search**: Debounced search with instant results
- **Responsive Design**: CSS Grid and Flexbox for all screen sizes
- **Filter System**: Multiple filter combinations
- **Result Preview**: Markdown content preview

## File Processing

### ChatGPT Custom Instructions
Uses the existing `gptparser.py` module to parse GPT files with structured metadata:
- GPT URL, Title, Description
- Instructions content
- Actions and Knowledge Base files
- Version tracking

### Markdown Files (Guides, Articles, etc.)
Automatic processing extracts:
- Title from headers or filename
- Description from content preview
- Tags from content analysis
- Category from directory structure

### Auto-Discovery
The system automatically:
- Scans for new files on refresh
- Maintains file modification tracking
- Updates search index in real-time
- Preserves existing file naming conventions

## Advanced Features (Optional)

### Embedding-Based Semantic Search
For enhanced search capabilities, you can add semantic search:

```bash
pip install sentence-transformers
```

This enables:
- Meaning-based search beyond keyword matching
- Related content discovery
- Semantic similarity scoring

### LLM Integration
Optional LLM connector for:
- Prompt enhancement suggestions
- Content summarization
- Tag generation improvements

## Development

### Project Structure
```
web_interface/
├── backend/
│   └── app.py              # FastAPI application
├── requirements.txt        # Python dependencies
├── start_server.py        # Startup script
└── README.md              # This file
```

### Adding New Content Types
To index additional file types or directories:

1. Extend `IndexManager.process_markdown_files()` in `app.py`
2. Add new category to the processing loop
3. Implement custom parsing logic if needed

### Customizing Search
The search algorithm can be tuned by modifying scoring weights in `IndexManager.search()`:
- Title matches: Currently weighted 10x
- Description matches: 5x
- Tag matches: 3x  
- Content matches: 1x

## Performance

- **Index Size**: ~1,800 items indexed in memory
- **Search Speed**: Sub-100ms response times
- **Memory Usage**: ~50MB for full index
- **Startup Time**: 2-3 seconds for initial indexing

## Troubleshooting

### Common Issues

**Import Error for gptparser**
- Ensure you're running from the correct directory
- The script automatically adds `.scripts/` to Python path

**No Results Found**
- Click "Refresh Index" to rebuild the search index
- Check that repository content exists in expected directories

**Server Won't Start**
- Verify Python 3.8+ is installed
- Install requirements: `pip install -r requirements.txt`
- Check port 8000 is not in use

### Debug Mode
Start with debug logging:
```bash
cd backend
python -c "import app; app.index_manager.build_index()"
```

## Contributing

The web interface is designed to automatically adapt to repository structure changes. When adding new content:

1. Follow existing file naming conventions
2. Use standard markdown formatting
3. Click "Refresh Index" in the web interface
4. New content will be automatically discoverable

## License

Same as the main repository - educational and research purposes.