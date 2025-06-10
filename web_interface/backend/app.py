"""
Web Interface Backend for TheBigEverythingPromptLibrary
FastAPI backend serving the prompt repository content with search capabilities.
"""

import os
import sys
import json
import re
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add the scripts directory to path to import gptparser
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT / ".scripts"))

import gptparser

app = FastAPI(title="Prompt Library API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptItem(BaseModel):
    id: str
    title: str
    description: str
    category: str
    subcategory: Optional[str] = None
    url: Optional[str] = None
    file_path: str
    content: str
    tags: List[str] = []
    created_date: Optional[str] = None
    version: Optional[str] = None

class SearchResponse(BaseModel):
    items: List[PromptItem]
    total: int
    query: str
    filters: Dict[str, Any]

class IndexManager:
    """Manages the search index for all prompt content"""
    
    def __init__(self):
        self.repo_root = REPO_ROOT
        self.index: List[PromptItem] = []
        self.last_updated = None
        self.build_index()
    
    def get_file_content(self, file_path: Path) -> str:
        """Read file content safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""
    
    def extract_markdown_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from markdown files"""
        metadata = {}
        lines = content.split('\n')
        
        # Look for key-value pairs at the beginning
        for line in lines[:20]:  # Check first 20 lines
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key and value:
                    metadata[key] = value
        
        return metadata
    
    def process_gpt_files(self) -> List[PromptItem]:
        """Process ChatGPT custom instruction files"""
        items = []
        
        for ok, gpt in gptparser.enum_gpts():
            if not ok:
                continue
                
            gpt_id = gpt.id()
            if not gpt_id:
                continue
            
            # Extract tags from description and title
            tags = []
            description = gpt.get('description', '') or ''
            title = gpt.get('title', '') or ''
            
            # Simple tag extraction from common keywords
            content_text = f"{title} {description} {gpt.get('instructions', '') or ''}"
            tag_keywords = ['coding', 'writing', 'analysis', 'creative', 'business', 'education', 
                          'productivity', 'gaming', 'art', 'security', 'jailbreak', 'assistant']
            
            for keyword in tag_keywords:
                if keyword.lower() in content_text.lower():
                    tags.append(keyword)
            
            item = PromptItem(
                id=gpt_id.id,
                title=title,
                description=description,
                category="CustomInstructions",
                subcategory="ChatGPT",
                url=gpt.get('url', ''),
                file_path=gpt.filename,
                content=gpt.get('instructions', '') or '',
                tags=tags,
                version=gpt.get('version', '')
            )
            items.append(item)
        
        return items
    
    def process_markdown_files(self, directory: str, category: str) -> List[PromptItem]:
        """Process markdown files from various directories"""
        items = []
        dir_path = self.repo_root / directory
        
        if not dir_path.exists():
            return items
        
        for file_path in dir_path.rglob("*.md"):
            if file_path.name == "README.md":
                continue
                
            content = self.get_file_content(file_path)
            if not content:
                continue
            
            metadata = self.extract_markdown_metadata(content)
            
            # Extract title from first header or filename
            title = metadata.get('title', '')
            if not title:
                for line in content.split('\n')[:10]:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
            if not title:
                title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
            
            # Extract description from content
            description = metadata.get('description', '')
            if not description:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('# ') and i + 1 < len(lines):
                        # Get the paragraph after the title
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if lines[j].strip() and not lines[j].startswith('#'):
                                description = lines[j].strip()[:200]
                                break
                        break
            
            # Generate tags from content
            tags = []
            content_lower = content.lower()
            tag_keywords = ['guide', 'tutorial', 'security', 'jailbreak', 'system', 'prompt', 
                          'engineering', 'ai', 'chatgpt', 'claude', 'cursor', 'analysis']
            
            for keyword in tag_keywords:
                if keyword in content_lower:
                    tags.append(keyword)
            
            # Add category-specific tags
            if category == "Guides":
                tags.append("guide")
            elif category == "Security":
                tags.append("security")
            elif category == "SystemPrompts":
                tags.append("system-prompt")
            
            item = PromptItem(
                id=f"{category}_{file_path.stem}",
                title=title,
                description=description,
                category=category,
                subcategory=str(file_path.parent.name) if file_path.parent.name != directory else None,
                file_path=str(file_path),
                content=content[:2000],  # Limit content for search
                tags=tags,
                created_date=metadata.get('date', '')
            )
            items.append(item)
        
        return items
    
    def build_index(self):
        """Build the complete search index"""
        print("Building search index...")
        self.index = []
        
        # Process ChatGPT custom instructions
        print("Processing GPT files...")
        self.index.extend(self.process_gpt_files())
        
        # Process other categories
        categories = [
            ("Guides", "Guides"),
            ("Articles", "Articles"), 
            ("SystemPrompts", "SystemPrompts"),
            ("Security", "Security"),
            ("Jailbreak", "Jailbreak")
        ]
        
        for directory, category in categories:
            print(f"Processing {category}...")
            self.index.extend(self.process_markdown_files(directory, category))
        
        self.last_updated = datetime.now()
        print(f"Index built with {len(self.index)} items")
    
    def search(self, query: str = "", category: str = "", tags: List[str] = None, 
               limit: int = 50, offset: int = 0) -> SearchResponse:
        """Search the index with filters"""
        if tags is None:
            tags = []
        
        results = self.index
        
        # Filter by category
        if category:
            results = [item for item in results if item.category.lower() == category.lower()]
        
        # Filter by tags
        if tags:
            results = [item for item in results 
                      if any(tag.lower() in [t.lower() for t in item.tags] for tag in tags)]
        
        # Search by query
        if query:
            query_lower = query.lower()
            scored_results = []
            
            for item in results:
                score = 0
                
                # Title match (highest weight)
                if query_lower in item.title.lower():
                    score += 10
                
                # Description match
                if query_lower in item.description.lower():
                    score += 5
                
                # Content match
                if query_lower in item.content.lower():
                    score += 1
                
                # Tag match
                for tag in item.tags:
                    if query_lower in tag.lower():
                        score += 3
                
                if score > 0:
                    scored_results.append((score, item))
            
            # Sort by score and extract items
            scored_results.sort(key=lambda x: x[0], reverse=True)
            results = [item for score, item in scored_results]
        
        # Apply pagination
        total = len(results)
        results = results[offset:offset + limit]
        
        return SearchResponse(
            items=results,
            total=total,
            query=query,
            filters={"category": category, "tags": tags}
        )
    
    def get_categories(self) -> Dict[str, int]:
        """Get all categories with counts"""
        categories = {}
        for item in self.index:
            categories[item.category] = categories.get(item.category, 0) + 1
        return categories
    
    def get_tags(self) -> Dict[str, int]:
        """Get all tags with counts"""
        tags = {}
        for item in self.index:
            for tag in item.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return tags

# Initialize the index manager
index_manager = IndexManager()

@app.get("/")
async def read_root():
    """Serve the frontend HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>The Big Everything Prompt Library</title>
        <script src="https://unpkg.com/marked/marked.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .search-section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .search-input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 16px; margin-bottom: 15px; }
            .filters { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }
            .filter-select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; background: white; }
            .results { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .result-item { padding: 20px; border-bottom: 1px solid #eee; }
            .result-item:last-child { border-bottom: none; }
            .result-title { font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2563eb; }
            .result-meta { font-size: 12px; color: #666; margin-bottom: 8px; }
            .result-description { margin-bottom: 10px; }
            .result-tags { display: flex; gap: 5px; flex-wrap: wrap; }
            .tag { background: #e5e7eb; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
            .loading { text-align: center; padding: 40px; color: #666; }
            .stats { font-size: 14px; color: #666; margin-bottom: 15px; }
            .refresh-btn { background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
            @media (max-width: 768px) { .filters { flex-direction: column; } .filter-select { width: 100%; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 The Big Everything Prompt Library</h1>
                <p>Search through 1,800+ prompts, guides, and AI resources</p>
            </div>
            
            <div class="search-section">
                <input type="text" id="searchInput" class="search-input" placeholder="Search prompts, guides, articles...">
                
                <div class="filters">
                    <select id="categoryFilter" class="filter-select">
                        <option value="">All Categories</option>
                    </select>
                    <select id="tagFilter" class="filter-select">
                        <option value="">All Tags</option>
                    </select>
                    <button id="refreshBtn" class="refresh-btn">Refresh Index</button>
                </div>
                
                <div id="stats" class="stats"></div>
            </div>
            
            <div id="results" class="results">
                <div class="loading">Loading prompts...</div>
            </div>
        </div>

        <script>
            let searchTimeout;
            let currentResults = [];
            
            // API calls
            async function searchPrompts(query = '', category = '', tag = '') {
                const params = new URLSearchParams();
                if (query) params.append('query', query);
                if (category) params.append('category', category);
                if (tag) params.append('tags', tag);
                
                const response = await fetch(`/api/search?${params}`);
                return await response.json();
            }
            
            async function getCategories() {
                const response = await fetch('/api/categories');
                return await response.json();
            }
            
            async function getTags() {
                const response = await fetch('/api/tags');
                return await response.json();
            }
            
            async function refreshIndex() {
                const response = await fetch('/api/refresh', { method: 'POST' });
                return await response.json();
            }
            
            // UI functions
            function renderResults(data) {
                const resultsDiv = document.getElementById('results');
                const statsDiv = document.getElementById('stats');
                
                statsDiv.textContent = `Found ${data.total} results`;
                
                if (data.items.length === 0) {
                    resultsDiv.innerHTML = '<div class="loading">No results found</div>';
                    return;
                }
                
                const html = data.items.map(item => `
                    <div class="result-item">
                        <div class="result-title">${escapeHtml(item.title)}</div>
                        <div class="result-meta">
                            ${item.category}${item.subcategory ? ` > ${item.subcategory}` : ''}
                            ${item.version ? ` (v${item.version})` : ''}
                        </div>
                        <div class="result-description">${escapeHtml(item.description)}</div>
                        <div class="result-tags">
                            ${item.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}
                        </div>
                    </div>
                `).join('');
                
                resultsDiv.innerHTML = html;
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            async function populateFilters() {
                const [categories, tags] = await Promise.all([getCategories(), getTags()]);
                
                const categorySelect = document.getElementById('categoryFilter');
                Object.entries(categories).forEach(([cat, count]) => {
                    const option = document.createElement('option');
                    option.value = cat;
                    option.textContent = `${cat} (${count})`;
                    categorySelect.appendChild(option);
                });
                
                const tagSelect = document.getElementById('tagFilter');
                Object.entries(tags)
                    .sort((a, b) => b[1] - a[1])  // Sort by count
                    .slice(0, 20)  // Top 20 tags
                    .forEach(([tag, count]) => {
                        const option = document.createElement('option');
                        option.value = tag;
                        option.textContent = `${tag} (${count})`;
                        tagSelect.appendChild(option);
                    });
            }
            
            function performSearch() {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(async () => {
                    const query = document.getElementById('searchInput').value;
                    const category = document.getElementById('categoryFilter').value;
                    const tag = document.getElementById('tagFilter').value;
                    
                    try {
                        const results = await searchPrompts(query, category, tag);
                        renderResults(results);
                    } catch (error) {
                        console.error('Search error:', error);
                        document.getElementById('results').innerHTML = '<div class="loading">Search error occurred</div>';
                    }
                }, 300);
            }
            
            // Event listeners
            document.getElementById('searchInput').addEventListener('input', performSearch);
            document.getElementById('categoryFilter').addEventListener('change', performSearch);
            document.getElementById('tagFilter').addEventListener('change', performSearch);
            
            document.getElementById('refreshBtn').addEventListener('click', async () => {
                const btn = document.getElementById('refreshBtn');
                btn.textContent = 'Refreshing...';
                btn.disabled = true;
                
                try {
                    await refreshIndex();
                    await populateFilters();
                    performSearch();
                } catch (error) {
                    console.error('Refresh error:', error);
                } finally {
                    btn.textContent = 'Refresh Index';
                    btn.disabled = false;
                }
            });
            
            // Initialize
            async function init() {
                await populateFilters();
                performSearch();
            }
            
            init();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/search", response_model=SearchResponse)
async def search_prompts(
    query: str = Query("", description="Search query"),
    category: str = Query("", description="Filter by category"),
    tags: str = Query("", description="Comma-separated tags"),
    limit: int = Query(50, description="Number of results per page"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Search prompts with optional filters"""
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    return index_manager.search(query, category, tag_list, limit, offset)

@app.get("/api/categories")
async def get_categories():
    """Get all available categories with counts"""
    return index_manager.get_categories()

@app.get("/api/tags")
async def get_tags():
    """Get all available tags with counts"""
    return index_manager.get_tags()

@app.post("/api/refresh")
async def refresh_index():
    """Refresh the search index"""
    index_manager.build_index()
    return {"status": "success", "message": f"Index refreshed with {len(index_manager.index)} items"}

@app.get("/api/item/{item_id}")
async def get_item(item_id: str):
    """Get detailed information about a specific item"""
    for item in index_manager.index:
        if item.id == item_id:
            # Return full content for the item
            full_content = index_manager.get_file_content(Path(item.file_path))
            return {**item.dict(), "full_content": full_content}
    
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)