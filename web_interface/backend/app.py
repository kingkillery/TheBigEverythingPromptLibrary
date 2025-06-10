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

# Enhanced search capabilities
try:
    from enhanced_search import create_enhanced_search_engine
    ENHANCED_SEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_SEARCH_AVAILABLE = False
    print("Enhanced search not available. Install with: pip install fuzzywuzzy python-levenshtein scikit-learn")

# Optional semantic search
try:
    from semantic_search import create_semantic_search_engine
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    print("Semantic search not available. Install with: pip install sentence-transformers numpy")

# Optional LLM connector
try:
    from llm_connector import create_llm_connector, enhance_prompt_workflow
    LLM_CONNECTOR_AVAILABLE = True
except ImportError:
    LLM_CONNECTOR_AVAILABLE = False
    print("LLM connector not available. Install with: pip install httpx")

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
        self.semantic_search = None
        self.llm_connector = None
        self.enhanced_search = None
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
        
        # Initialize semantic search if available
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                self.semantic_search = create_semantic_search_engine()
                if self.semantic_search:
                    self.semantic_search.build_embeddings(self.index)
            except Exception as e:
                print(f"Could not initialize semantic search: {e}")
                self.semantic_search = None
        
        # Initialize LLM connector if available
        if LLM_CONNECTOR_AVAILABLE:
            try:
                self.llm_connector = create_llm_connector()
                if self.llm_connector:
                    print(f"✅ LLM connector initialized with {len(self.llm_connector.get_available_models())} free models")
            except Exception as e:
                print(f"Could not initialize LLM connector: {e}")
                self.llm_connector = None
        
        # Initialize enhanced search if available
        if ENHANCED_SEARCH_AVAILABLE:
            try:
                self.enhanced_search = create_enhanced_search_engine()
                print("✅ Enhanced search engine initialized")
            except Exception as e:
                print(f"Could not initialize enhanced search: {e}")
                self.enhanced_search = None
    
    def search(self, query: str = "", category: str = "", tags: List[str] = None, 
               limit: int = 50, offset: int = 0, min_quality: float = 0.0,
               sort_by: str = "relevance") -> SearchResponse:
        """Enhanced search with quality filtering and advanced scoring"""
        if tags is None:
            tags = []
        
        # Use enhanced search if available
        if self.enhanced_search and (query or min_quality > 0.0):
            results, stats = self.enhanced_search.advanced_search(
                items=self.index,
                query=query,
                min_quality=min_quality,
                max_results=limit + offset,  # Get more for pagination
                category_filter=category,
                tag_filter=tags,
                sort_by=sort_by
            )
            
            # Apply pagination
            total = len(results)
            paginated_results = results[offset:offset + limit]
            
            return SearchResponse(
                items=paginated_results,
                total=total,
                query=query,
                filters={"category": category, "tags": tags, "min_quality": min_quality, 
                        "sort_by": sort_by, "search_stats": stats}
            )
        
        # Fallback to original search logic
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
            if self.semantic_search and len(query.split()) > 1:
                # Use hybrid search for multi-word queries
                keyword_results = self._keyword_search(results, query)
                results = self.semantic_search.hybrid_search(query, keyword_results)
            else:
                # Use keyword search for single words or when semantic search unavailable
                results = self._keyword_search(results, query)
        
        # Apply pagination
        total = len(results)
        results = results[offset:offset + limit]
        
        return SearchResponse(
            items=results,
            total=total,
            query=query,
            filters={"category": category, "tags": tags}
        )
    
    def _keyword_search(self, items: List[PromptItem], query: str) -> List[PromptItem]:
        """Perform keyword-based search"""
        query_lower = query.lower()
        scored_results = []
        
        for item in items:
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
        return [item for score, item in scored_results]
    
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
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .search-section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .search-input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 16px; margin-bottom: 15px; }
            .filters { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 15px; }
            .filter-select { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; background: white; }
            .quality-slider { width: 100%; margin: 10px 0; }
            .advanced-filters { display: none; margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 6px; }
            .toggle-filters { color: #2563eb; cursor: pointer; text-decoration: underline; font-size: 14px; }
            .results { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .result-item { padding: 20px; border-bottom: 1px solid #eee; position: relative; }
            .result-item:last-child { border-bottom: none; }
            .result-item.high-quality { border-left: 4px solid #10b981; }
            .result-item.medium-quality { border-left: 4px solid #f59e0b; }
            .result-item.low-quality { border-left: 4px solid #ef4444; }
            .result-title { font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2563eb; }
            .result-meta { font-size: 12px; color: #666; margin-bottom: 8px; display: flex; gap: 15px; align-items: center; }
            .result-description { margin-bottom: 10px; }
            .result-tags { display: flex; gap: 5px; flex-wrap: wrap; }
            .tag { background: #e5e7eb; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
            .quality-badge { padding: 2px 6px; border-radius: 10px; font-size: 11px; font-weight: 600; }
            .quality-high { background: #d1fae5; color: #065f46; }
            .quality-medium { background: #fef3c7; color: #92400e; }
            .quality-low { background: #fee2e2; color: #991b1b; }
            .search-stats { background: #f3f4f6; padding: 10px; border-radius: 6px; margin-bottom: 15px; font-size: 14px; }
            .loading { text-align: center; padding: 40px; color: #666; }
            .stats { font-size: 14px; color: #666; margin-bottom: 15px; }
            .refresh-btn { background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
            .llm-btn { background: #10b981; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 5px; }
            .llm-section { background: #f9f9f9; padding: 15px; margin-top: 10px; border-radius: 6px; border-left: 4px solid #10b981; }
            .llm-result { background: white; padding: 10px; border-radius: 4px; margin: 10px 0; white-space: pre-wrap; }
            .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }
            .modal-content { background: white; margin: 5% auto; padding: 20px; border-radius: 8px; max-width: 800px; max-height: 80%; overflow-y: auto; }
            @media (max-width: 768px) { .filters { flex-direction: column; } .filter-select { width: 100%; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 The Big Everything Prompt Library</h1>
                <p>Search through 1,800+ high-quality prompts, guides, and AI resources with advanced filtering</p>
                <div style="font-size: 14px; color: #666; margin-top: 10px;">
                    ✨ Enhanced with quality scoring, fuzzy search, and semantic matching
                </div>
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
                    <select id="sortFilter" class="filter-select">
                        <option value="relevance">Sort: Relevance</option>
                        <option value="quality">Sort: Quality</option>
                        <option value="title">Sort: Title</option>
                        <option value="newest">Sort: Newest</option>
                    </select>
                    <button id="refreshBtn" class="refresh-btn">Refresh Index</button>
                    <span class="toggle-filters" onclick="toggleAdvancedFilters()">🔧 Advanced Filters</span>
                </div>
                
                <div id="advancedFilters" class="advanced-filters">
                    <label for="qualitySlider">Minimum Quality Score: <span id="qualityValue">0.0</span></label>
                    <input type="range" id="qualitySlider" class="quality-slider" min="0" max="1" step="0.1" value="0">
                    <div style="margin-top: 10px;">
                        <label><input type="checkbox" id="highQualityOnly"> High Quality Only (0.7+)</label>
                    </div>
                </div>
                
                <div id="stats" class="stats"></div>
                <div id="searchStats" class="search-stats" style="display: none;"></div>
            </div>
            
            <div id="results" class="results">
                <div class="loading">Loading prompts...</div>
            </div>
        </div>
        
        <!-- LLM Enhancement Modal -->
        <div id="llmModal" class="modal">
            <div class="modal-content">
                <h3>🤖 AI Enhancement Tools</h3>
                <div id="llmContent"></div>
                <button onclick="closeLLMModal()" style="margin-top: 20px; padding: 8px 16px; background: #6b7280; color: white; border: none; border-radius: 4px; cursor: pointer;">Close</button>
            </div>
        </div>

        <script>
            let searchTimeout;
            let currentResults = [];
            
            // API calls
            async function searchPrompts(query = '', category = '', tag = '', minQuality = 0.0, sortBy = 'relevance') {
                const params = new URLSearchParams();
                if (query) params.append('query', query);
                if (category) params.append('category', category);
                if (tag) params.append('tags', tag);
                if (minQuality > 0) params.append('min_quality', minQuality);
                if (sortBy !== 'relevance') params.append('sort_by', sortBy);
                
                const response = await fetch(`/api/search?${params}`);
                return await response.json();
            }
            
            function toggleAdvancedFilters() {
                const filters = document.getElementById('advancedFilters');
                filters.style.display = filters.style.display === 'none' ? 'block' : 'none';
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
            
            // LLM API calls
            async function enhancePrompt(prompt, type = 'improve') {
                const response = await fetch('/api/llm/enhance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, type })
                });
                return await response.json();
            }
            
            async function analyzePrompt(prompt, includeAll = false) {
                const response = await fetch('/api/llm/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, include_all: includeAll })
                });
                return await response.json();
            }
            
            async function generateTags(prompt, title = '') {
                const response = await fetch('/api/llm/generate-tags', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, title })
                });
                return await response.json();
            }
            
            async function checkLLMAvailable() {
                try {
                    const response = await fetch('/api/llm/models');
                    return response.ok;
                } catch {
                    return false;
                }
            }
            
            // UI functions
            function renderResults(data) {
                const resultsDiv = document.getElementById('results');
                const statsDiv = document.getElementById('stats');
                const searchStatsDiv = document.getElementById('searchStats');
                
                statsDiv.textContent = `Found ${data.total} results`;
                
                // Show enhanced search stats if available
                if (data.filters && data.filters.search_stats) {
                    const stats = data.filters.search_stats;
                    searchStatsDiv.innerHTML = `
                        <strong>Search Analysis:</strong> 
                        Average Quality: ${(stats.avg_quality * 100).toFixed(0)}% | 
                        High Quality: ${stats.quality_distribution.high} items | 
                        Keywords: ${stats.query_keywords.join(', ')}
                    `;
                    searchStatsDiv.style.display = 'block';
                } else {
                    searchStatsDiv.style.display = 'none';
                }
                
                if (data.items.length === 0) {
                    resultsDiv.innerHTML = '<div class="loading">No results found. Try adjusting your filters or search terms.</div>';
                    return;
                }
                
                const html = data.items.map(item => {
                    // Calculate quality score for display (if enhanced search is available)
                    const qualityClass = getQualityClass(item);
                    const qualityBadge = getQualityBadge(item);
                    
                    return `
                        <div class="result-item ${qualityClass}">
                            <div class="result-title">${escapeHtml(item.title)}</div>
                            <div class="result-meta">
                                <span>${item.category}${item.subcategory ? ` > ${item.subcategory}` : ''}</span>
                                <span>${item.version ? `v${item.version}` : ''}</span>
                                ${qualityBadge}
                                <span id="llm-buttons-${item.id}"></span>
                            </div>
                            <div class="result-description">${escapeHtml(item.description)}</div>
                            <div class="result-tags">
                                ${item.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}
                            </div>
                        </div>
                    `;
                }).join('');
                
                resultsDiv.innerHTML = html;
                
                // Add LLM buttons if available
                addLLMButtons(data.items);
            }
            
            function getQualityClass(item) {
                // Simple heuristic for quality classification
                const contentLength = item.content.length;
                const hasStructure = item.content.includes('#') || item.content.includes('1.') || item.content.includes('-');
                const detailedDescription = item.description.length > 50;
                
                if (contentLength > 800 && hasStructure && detailedDescription) return 'high-quality';
                if (contentLength > 300 && (hasStructure || detailedDescription)) return 'medium-quality';
                return 'low-quality';
            }
            
            function getQualityBadge(item) {
                const qualityClass = getQualityClass(item);
                if (qualityClass === 'high-quality') return '<span class="quality-badge quality-high">High Quality</span>';
                if (qualityClass === 'medium-quality') return '<span class="quality-badge quality-medium">Medium Quality</span>';
                return '<span class="quality-badge quality-low">Basic</span>';
            }
            
            async function addLLMButtons(items) {
                const llmAvailable = await checkLLMAvailable();
                if (!llmAvailable) return;
                
                items.forEach(item => {
                    const buttonContainer = document.getElementById(`llm-buttons-${item.id}`);
                    if (buttonContainer) {
                        buttonContainer.innerHTML = `
                            <button class="llm-btn" onclick="showLLMEnhancement('${item.id}', '${escapeHtml(item.title)}', '${escapeHtml(item.content).substring(0, 500)}')">
                                🤖 Enhance
                            </button>
                        `;
                    }
                });
            }
            
            function showLLMEnhancement(itemId, title, content) {
                const modal = document.getElementById('llmModal');
                const modalContent = document.getElementById('llmContent');
                
                modalContent.innerHTML = `
                    <h4>Enhancing: ${title}</h4>
                    <div class="llm-section">
                        <button onclick="performLLMAction('${itemId}', '${escapeHtml(content)}', 'analyze')" class="llm-btn">
                            🔍 Analyze Prompt
                        </button>
                        <button onclick="performLLMAction('${itemId}', '${escapeHtml(content)}', 'improve')" class="llm-btn">
                            ✨ Improve
                        </button>
                        <button onclick="performLLMAction('${itemId}', '${escapeHtml(content)}', 'variants')" class="llm-btn">
                            🔄 Create Variants
                        </button>
                        <button onclick="performLLMAction('${itemId}', '${escapeHtml(content)}', 'tags')" class="llm-btn">
                            🏷️ Generate Tags
                        </button>
                    </div>
                    <div id="llm-results-${itemId}"></div>
                `;
                
                modal.style.display = 'block';
            }
            
            async function performLLMAction(itemId, content, action) {
                const resultsDiv = document.getElementById(`llm-results-${itemId}`);
                resultsDiv.innerHTML = '<div class="loading">🤖 AI is working...</div>';
                
                try {
                    let result;
                    
                    switch(action) {
                        case 'analyze':
                            result = await analyzePrompt(content, true);
                            displayAnalysisResult(resultsDiv, result);
                            break;
                        case 'improve':
                            result = await enhancePrompt(content, 'improve');
                            displayEnhancementResult(resultsDiv, result);
                            break;
                        case 'variants':
                            result = await enhancePrompt(content, 'variants');
                            displayEnhancementResult(resultsDiv, result);
                            break;
                        case 'tags':
                            result = await generateTags(content);
                            displayTagsResult(resultsDiv, result);
                            break;
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="llm-result">❌ Error: ${error.message}</div>`;
                }
            }
            
            function displayAnalysisResult(container, result) {
                let html = '<div class="llm-section"><h5>🔍 AI Analysis Results:</h5>';
                
                if (result.enhancement) {
                    html += `<div class="llm-result"><strong>Enhancement Suggestions:</strong><br>${result.enhancement.enhanced_content}</div>`;
                }
                
                if (result.suggested_tags) {
                    html += `<div class="llm-result"><strong>Suggested Tags:</strong><br>${result.suggested_tags.join(', ')}</div>`;
                }
                
                if (result.detected_type) {
                    html += `<div class="llm-result"><strong>Detected Type:</strong> ${result.detected_type}</div>`;
                }
                
                if (result.use_cases) {
                    html += `<div class="llm-result"><strong>Use Cases:</strong><br>• ${result.use_cases.join('<br>• ')}</div>`;
                }
                
                html += '</div>';
                container.innerHTML = html;
            }
            
            function displayEnhancementResult(container, result) {
                if (result && result.enhanced_content) {
                    container.innerHTML = `
                        <div class="llm-section">
                            <h5>✨ Enhanced Version:</h5>
                            <div class="llm-result">${result.enhanced_content}</div>
                            <small>Generated by: ${result.model_used}</small>
                        </div>
                    `;
                } else {
                    container.innerHTML = '<div class="llm-result">❌ Enhancement failed</div>';
                }
            }
            
            function displayTagsResult(container, result) {
                if (result && result.tags) {
                    const tagsHtml = result.tags.map(tag => `<span class="tag">${tag}</span>`).join(' ');
                    container.innerHTML = `
                        <div class="llm-section">
                            <h5>🏷️ Generated Tags:</h5>
                            <div class="llm-result">${tagsHtml}</div>
                        </div>
                    `;
                } else {
                    container.innerHTML = '<div class="llm-result">❌ Tag generation failed</div>';
                }
            }
            
            function closeLLMModal() {
                document.getElementById('llmModal').style.display = 'none';
            }
            
            // Close modal when clicking outside
            window.onclick = function(event) {
                const modal = document.getElementById('llmModal');
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
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
                    const sortBy = document.getElementById('sortFilter').value;
                    
                    // Get quality filters
                    let minQuality = 0.0;
                    const qualitySlider = document.getElementById('qualitySlider');
                    const highQualityOnly = document.getElementById('highQualityOnly');
                    
                    if (qualitySlider) {
                        minQuality = parseFloat(qualitySlider.value);
                    }
                    if (highQualityOnly && highQualityOnly.checked) {
                        minQuality = Math.max(minQuality, 0.7);
                    }
                    
                    try {
                        const results = await searchPrompts(query, category, tag, minQuality, sortBy);
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
            document.getElementById('sortFilter').addEventListener('change', performSearch);
            
            // Quality filter listeners
            document.addEventListener('DOMContentLoaded', function() {
                const qualitySlider = document.getElementById('qualitySlider');
                const qualityValue = document.getElementById('qualityValue');
                const highQualityOnly = document.getElementById('highQualityOnly');
                
                if (qualitySlider) {
                    qualitySlider.addEventListener('input', function() {
                        qualityValue.textContent = this.value;
                        performSearch();
                    });
                }
                
                if (highQualityOnly) {
                    highQualityOnly.addEventListener('change', performSearch);
                }
            });
            
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
    offset: int = Query(0, description="Offset for pagination"),
    min_quality: float = Query(0.0, description="Minimum quality score (0.0-1.0)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, quality, title, newest")
):
    """Enhanced search with quality filtering and advanced scoring"""
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    return index_manager.search(query, category, tag_list, limit, offset, min_quality, sort_by)

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

@app.get("/api/similar/{item_id}")
async def get_similar_items(item_id: str, limit: int = Query(10, description="Number of similar items")):
    """Get items similar to the specified item (requires semantic search)"""
    if not index_manager.semantic_search:
        raise HTTPException(status_code=501, detail="Semantic search not available")
    
    similar_items = index_manager.semantic_search.find_similar(item_id, limit)
    return {
        "item_id": item_id,
        "similar_items": [{"similarity": score, "item": item.dict()} for score, item in similar_items]
    }

@app.get("/api/semantic-search")
async def semantic_search_endpoint(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, description="Number of results")
):
    """Pure semantic search endpoint"""
    if not index_manager.semantic_search:
        raise HTTPException(status_code=501, detail="Semantic search not available")
    
    results = index_manager.semantic_search.semantic_search(query, limit)
    return {
        "query": query,
        "results": [{"similarity": score, "item": item.dict()} for score, item in results]
    }

# LLM Enhancement Endpoints
@app.get("/api/llm/models")
async def get_llm_models():
    """Get available LLM models"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    return index_manager.llm_connector.get_available_models()

@app.post("/api/llm/enhance")
async def enhance_prompt_endpoint(
    request: dict
):
    """Enhance a prompt using LLM"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    prompt_content = request.get("prompt", "")
    enhancement_type = request.get("type", "improve")
    
    if not prompt_content:
        raise HTTPException(status_code=400, detail="Prompt content required")
    
    result = await index_manager.llm_connector.enhance_prompt(prompt_content, enhancement_type)
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Enhancement failed")

@app.post("/api/llm/analyze")
async def analyze_prompt_endpoint(
    request: dict
):
    """Analyze a prompt comprehensively"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    prompt_content = request.get("prompt", "")
    include_all = request.get("include_all", False)
    
    if not prompt_content:
        raise HTTPException(status_code=400, detail="Prompt content required")
    
    result = await enhance_prompt_workflow(index_manager.llm_connector, prompt_content, include_all)
    return result

@app.post("/api/llm/generate-tags")
async def generate_tags_endpoint(
    request: dict
):
    """Generate tags for a prompt"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    prompt_content = request.get("prompt", "")
    title = request.get("title", "")
    
    if not prompt_content:
        raise HTTPException(status_code=400, detail="Prompt content required")
    
    tags = await index_manager.llm_connector.generate_tags(prompt_content, title)
    return {"tags": tags or []}

@app.post("/api/llm/summarize")
async def summarize_prompt_endpoint(
    request: dict
):
    """Summarize a prompt"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    prompt_content = request.get("prompt", "")
    max_length = request.get("max_length", 200)
    
    if not prompt_content:
        raise HTTPException(status_code=400, detail="Prompt content required")
    
    summary = await index_manager.llm_connector.summarize_prompt(prompt_content, max_length)
    return {"summary": summary or ""}

@app.post("/api/llm/compare")
async def compare_prompts_endpoint(
    request: dict
):
    """Compare two prompts"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    prompt1 = request.get("prompt1", "")
    prompt2 = request.get("prompt2", "")
    
    if not prompt1 or not prompt2:
        raise HTTPException(status_code=400, detail="Both prompts required")
    
    comparison = await index_manager.llm_connector.compare_prompts(prompt1, prompt2)
    return {"comparison": comparison or ""}

@app.post("/api/llm/use-cases")
async def suggest_use_cases_endpoint(
    request: dict
):
    """Suggest use cases for a prompt"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    prompt_content = request.get("prompt", "")
    
    if not prompt_content:
        raise HTTPException(status_code=400, detail="Prompt content required")
    
    use_cases = await index_manager.llm_connector.suggest_use_cases(prompt_content)
    return {"use_cases": use_cases or []}

@app.get("/api/search/suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="Partial search query")
):
    """Get search suggestions based on query"""
    if not index_manager.enhanced_search:
        return {"suggestions": []}
    
    suggestions = index_manager.enhanced_search.suggest_related_queries(query, index_manager.index)
    return {"suggestions": suggestions}

@app.get("/api/quality-filter")
async def get_quality_distribution():
    """Get quality score distribution for the entire collection"""
    if not index_manager.enhanced_search:
        return {"distribution": {}, "available": False}
    
    quality_scores = []
    for item in index_manager.index:
        score = index_manager.enhanced_search.quality_scorer.score_prompt_quality(item)
        quality_scores.append(score)
    
    import numpy as np
    distribution = {
        "total_items": len(quality_scores),
        "average_quality": float(np.mean(quality_scores)),
        "high_quality": len([s for s in quality_scores if s >= 0.7]),
        "medium_quality": len([s for s in quality_scores if 0.4 <= s < 0.7]),
        "low_quality": len([s for s in quality_scores if s < 0.4]),
        "percentiles": {
            "90th": float(np.percentile(quality_scores, 90)),
            "75th": float(np.percentile(quality_scores, 75)),
            "50th": float(np.percentile(quality_scores, 50)),
            "25th": float(np.percentile(quality_scores, 25))
        }
    }
    
    return {"distribution": distribution, "available": True}

@app.get("/api/search/analyze")
async def analyze_search_results(
    query: str = Query(..., description="Search query to analyze")
):
    """Analyze search result quality"""
    if not index_manager.enhanced_search:
        return {"analysis": {}, "available": False}
    
    # Perform search
    search_results = index_manager.search(query=query, limit=20)
    
    # Analyze results
    analysis = index_manager.enhanced_search.analyze_search_quality(search_results.items, query)
    
    return {"analysis": analysis, "available": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)