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

from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Optional trending RSS feed
# try:
#     from trending_feed import get_trending_feed
#     TRENDING_FEED_AVAILABLE = True
# except ImportError:
#     TRENDING_FEED_AVAILABLE = False
#     print("Trending feed not available. Install with: pip install pygooglenews")

# Add the scripts directory to path to import gptparser
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT / ".scripts"))
# Ensure backend directory is on import path for sibling modules like collections_db
sys.path.append(str(Path(__file__).parent))

import gptparser  # type: ignore

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

# Add collections_db helper
from collections_db import (
    create_collection,
    list_collections,
    add_item as db_add_item,
    remove_item as db_remove_item,
    get_collection_items,
    record_view,
    record_graft,
    get_popular,
    get_trending,
    get_most_grafted,
    get_new_sprouts,
    rename_collection as db_rename_collection,
    delete_collection as db_delete_collection,
    create_chain,
    list_chains,
    get_chain,
    update_chain,
    delete_chain,
    create_prompt_version,
    list_prompt_versions,
)

# Import category configuration
from category_config import (
    CATEGORIES,
    get_category_by_id,
    get_subcategory_by_id,
    suggest_category,
    get_all_categories,
    get_category_hierarchy
)

from rss_news import rss_fetcher

# Import chat API
try:
    from chat_api import router as chat_router
    CHAT_API_AVAILABLE = True
except ImportError:
    CHAT_API_AVAILABLE = False
    print("Chat API not available. Some AI provider dependencies may be missing.")

app = FastAPI(title="Prompt Library API", version="1.0.0")

# Include chat router if available
if CHAT_API_AVAILABLE:
    app.include_router(chat_router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template & static file support
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

class PromptItem(BaseModel):
    id: str
    title: str
    description: str
    category: str
    subcategory: Optional[str] = None
    meta_category: Optional[str] = None  # New hierarchical category
    meta_subcategory: Optional[str] = None  # New hierarchical subcategory
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

class ConfigUpdate(BaseModel):
    stop_words: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None

class ChatRequest(BaseModel):
    prompt: str
    max_results: int = 5
    format: str = "promptscript"  # Available: promptscript, yaml, plain

class ChatResponse(BaseModel):
    matches: List[PromptItem]
    optimized_prompt: str
    tweaked_match: str

# --- New model for setting API key ---
class ApiKeyRequest(BaseModel):
    api_key: str

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
            instructions = gpt.get('instructions', '') or ''
            
            # Combined text for analysis
            content_text = f"{title} {description} {instructions}"
            
            # Auto-categorize using our category configuration
            suggested_cat = suggest_category(content_text)
            meta_category = None
            meta_subcategory = None
            
            if suggested_cat:
                meta_category = suggested_cat[0]
                meta_subcategory = suggested_cat[1]
            
            # Enhanced tag extraction using category keywords
            all_keywords = set()
            
            # Add keywords from matched category
            if meta_category:
                cat = get_category_by_id(meta_category)
                if cat:
                    all_keywords.update(cat.keywords)
                    if meta_subcategory:
                        subcat = get_subcategory_by_id(meta_category, meta_subcategory)
                        if subcat:
                            all_keywords.update(subcat.keywords)
            
            # Simple tag extraction from common keywords
            tag_keywords = ['coding', 'writing', 'analysis', 'creative', 'business', 'education', 
                          'productivity', 'gaming', 'art', 'security', 'jailbreak', 'assistant']
            all_keywords.update(tag_keywords)
            
            for keyword in all_keywords:
                if keyword.lower() in content_text.lower():
                    tags.append(keyword)
            
            # Deduplicate tags
            tags = list(set(tags))[:10]  # Limit to 10 tags
            
            item = PromptItem(
                id=gpt_id.id,
                title=title,
                description=description,
                category="CustomInstructions",
                subcategory="ChatGPT",
                meta_category=meta_category,
                meta_subcategory=meta_subcategory,
                url=gpt.get('url', ''),
                file_path=gpt.filename,
                content=instructions,
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
            elif category == "DeepResearch":
                tags.append("research")
            
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
            ("Jailbreak", "Jailbreak"),
            ("DeepResearch", "DeepResearch"),
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
               sort_by: str = "relevance", meta_category: str = "", 
               meta_subcategory: str = "") -> SearchResponse:
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
                sort_by=sort_by,
                meta_category_filter=meta_category,
                meta_subcategory_filter=meta_subcategory
            )
            
            # Apply pagination
            total = len(results)
            paginated_results = results[offset:offset + limit]
            
            return SearchResponse(
                items=paginated_results,
                total=total,
                query=query,
                filters={"category": category, "tags": tags, "min_quality": min_quality, 
                        "sort_by": sort_by, "meta_category": meta_category, 
                        "meta_subcategory": meta_subcategory, "search_stats": stats}
            )
        
        # Fallback to original search logic
        results = self.index
        
        # Filter by category
        if category:
            results = [item for item in results if item.category.lower() == category.lower()]
        
        # Filter by meta-category
        if meta_category:
            results = [item for item in results if item.meta_category == meta_category]
            
        # Filter by meta-subcategory
        if meta_subcategory:
            results = [item for item in results if item.meta_subcategory == meta_subcategory]
        
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
            filters={"category": category, "tags": tags, "meta_category": meta_category, 
                    "meta_subcategory": meta_subcategory}
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
        """Get all categories with counts; include expected categories with zero items"""
        counts: Dict[str, int] = {c: 0 for c in getattr(self, "_expected_categories", [])}
        for item in self.index:
            counts[item.category] = counts.get(item.category, 0) + 1
        # Sort alphabetically for stable dropdown order
        return dict(sorted(counts.items()))
    
    def get_meta_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get all meta-categories with counts and metadata"""
        meta_counts = {}
        
        # Initialize with all categories from config
        for cat_id, category in CATEGORIES.items():
            meta_counts[cat_id] = {
                "name": category.name,
                "description": category.description,
                "count": 0,
                "icon": category.icon,
                "color": category.color,
                "subcategories": {}
            }
            
            # Initialize subcategory counts
            for subcat in category.subcategories:
                meta_counts[cat_id]["subcategories"][subcat.id] = {
                    "name": subcat.name,
                    "description": subcat.description,
                    "count": 0,
                    "icon": subcat.icon
                }
        
        # Count items in each category
        for item in self.index:
            if item.meta_category and item.meta_category in meta_counts:
                meta_counts[item.meta_category]["count"] += 1
                
                if item.meta_subcategory and item.meta_subcategory in meta_counts[item.meta_category]["subcategories"]:
                    meta_counts[item.meta_category]["subcategories"][item.meta_subcategory]["count"] += 1
        
        return meta_counts
    
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
async def read_root(request: Request):
    """Serve the frontend HTML"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the AI chat interface"""
    chat_html_path = Path(__file__).parent.parent / "chat_interface.html"
    if chat_html_path.exists():
        with open(chat_html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="<h1>Chat Interface Not Found</h1><p>The chat interface file is missing.</p>", status_code=404)

@app.get("/api/ping")
async def ping():
    """Health check and version info"""
    return {
        "status": "ok",
        "version": app.version,
        "items_indexed": len(index_manager.index)
    }

# ---------------------- Search Configuration Endpoints ----------------------

@app.get("/api/search/config")
async def get_search_config():
    """Get current search configuration (stop words & weights)"""
    if not index_manager.enhanced_search:
        raise HTTPException(status_code=501, detail="Enhanced search not available")
    return index_manager.enhanced_search.get_config()

@app.post("/api/search/config")
async def update_search_config(config: ConfigUpdate):
    """Update search configuration at runtime"""
    if not index_manager.enhanced_search:
        raise HTTPException(status_code=501, detail="Enhanced search not available")
    cfg_dict: Dict[str, Any] = {}
    if config.stop_words is not None:
        cfg_dict['stop_words'] = config.stop_words
    if config.weights is not None:
        cfg_dict['weights'] = config.weights
    if not cfg_dict:
        raise HTTPException(status_code=400, detail="No valid configuration fields provided")
    index_manager.enhanced_search.update_config(cfg_dict)
    return {"status": "updated", "config": index_manager.enhanced_search.get_config()}

# ---------------------- Detailed Search Endpoint ---------------------------

@app.get("/api/search/details")
async def search_prompts_details(
    query: str = Query("", description="Search query"),
    category: str = Query("", description="Filter by category"),
    tags: str = Query("", description="Comma-separated tags"),
    limit: int = Query(50, description="Number of results per page"),
    offset: int = Query(0, description="Offset for pagination"),
    min_quality: float = Query(0.0, description="Minimum quality score (0.0-1.0)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, quality, title, newest"),
    meta_category: str = Query("", description="Filter by meta-category"),
    meta_subcategory: str = Query("", description="Filter by meta-subcategory")
):
    """Search that returns per-item score breakdown for transparency"""
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

    if not index_manager.enhanced_search:
        raise HTTPException(status_code=501, detail="Enhanced search not available")

    # We cannot apply offset easily on detailed results after breakdown; so request larger result set then slice
    details, stats = index_manager.enhanced_search.advanced_search_details(
        items=index_manager.index,
        query=query,
        min_quality=min_quality,
        max_results=limit + offset,
        category_filter=category,
        tag_filter=tag_list,
        sort_by=sort_by,
        meta_category_filter=meta_category,
        meta_subcategory_filter=meta_subcategory
    )
    total = stats.get('total_found', len(details))
    paginated_details = details[offset:offset + limit]

    # Convert Pydantic PromptItem into dict to ensure JSON serialisable
    serialised = [
        {
            **{
                "item": d["item"].dict(),
                "score": d["score"],
                "breakdown": d["breakdown"]
            }
        } for d in paginated_details
    ]

    return {
        "details": serialised,
        "total": total,
        "query": query,
        "filters": {"category": category, "tags": tag_list, "min_quality": min_quality, "sort_by": sort_by},
        "search_stats": stats
    }

@app.get("/api/search", response_model=SearchResponse)
async def search_prompts(
    query: str = Query("", description="Search query"),
    category: str = Query("", description="Filter by category"),
    tags: str = Query("", description="Comma-separated tags"),
    limit: int = Query(50, description="Number of results per page"),
    offset: int = Query(0, description="Offset for pagination"),
    min_quality: float = Query(0.0, description="Minimum quality score (0.0-1.0)"),
    sort_by: str = Query("relevance", description="Sort by: relevance, quality, title, newest"),
    meta_category: str = Query("", description="Filter by meta-category"),
    meta_subcategory: str = Query("", description="Filter by meta-subcategory")
):
    """Enhanced search with quality filtering and advanced scoring"""
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    return index_manager.search(query, category, tag_list, limit, offset, min_quality, sort_by, meta_category, meta_subcategory)

@app.get("/api/categories")
async def get_categories():
    """Get all available categories with counts"""
    return index_manager.get_categories()

@app.get("/api/meta-categories")
async def get_meta_categories():
    """Get all meta-categories with counts and metadata"""
    return index_manager.get_meta_categories()

@app.get("/api/category-config")
async def get_category_config():
    """Get the full category configuration"""
    return get_all_categories()

@app.get("/api/category-hierarchy")
async def get_category_hierarchy_endpoint():
    """Get category hierarchy for dropdowns"""
    return get_category_hierarchy()

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
            # Record view stat
            try:
                record_view(item_id)
            except Exception:
                pass
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
        # Record graft stat
        try:
            record_graft(request.get("original_id", ""))
        except Exception:
            pass
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

@app.post("/api/llm/promptsmith-9000")
async def promptsmith_9000_generate_endpoint(
    request: dict
):
    """
    PromptSmith 9000 - Battle-tested prompt generation using Anthropic's best practices
    Transforms raw user input into production-ready prompts with library integration
    """
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    user_prompt = request.get("user_prompt", "")
    context = request.get("context", "")
    preferences = request.get("preferences", "")
    max_library_items = request.get("max_library_items", 10)
    
    if not user_prompt:
        raise HTTPException(status_code=400, detail="user_prompt is required")
    
    # Get relevant prompts from library for context
    prompt_library = []
    if user_prompt.strip():
        # Use existing search to find relevant prompts
        search_results = index_manager.search(
            query=user_prompt,
            category="",
            tags=[],
            limit=max_library_items,
            offset=0,
            min_quality=0.5,  # Only include decent quality prompts
            sort_by="relevance"
        )
        
        # Convert search results to library format
        for item in search_results.results:
            prompt_library.append({
                "title": item.title,
                "content": item.content[:1000],  # Truncate for context length
                "tags": item.tags,
                "quality_score": getattr(item, 'quality_score', 0.0),
                "category": item.category
            })
    
    try:
        result = await index_manager.llm_connector.promptsmith_9000_generate(
            user_prompt=user_prompt,
            prompt_library=prompt_library,
            context=context,
            preferences=preferences
        )
        
        if result and "error" not in result:
            return {
                "success": True,
                "result": result,
                "library_items_used": len(prompt_library),
                "workflow": "PromptSmith 9000"
            }
        else:
            error_message = result.get("error", "Unknown error") if result else "No result returned"
            raise HTTPException(status_code=500, detail=f"PromptSmith 9000 generation failed: {error_message}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PromptSmith 9000 error: {str(e)}")

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

# ---------------------- PromptScript Utilities ----------------------------

PROMPTSCRIPT_EXPLANATION = (
    "Below is a prompt written in PromptScript, a concise prompt-design language that encodes tasks, "
    "parameters, context, and output formats with compact symbols so agents can parse instructions deterministically. "
    "In PromptScript, ':' introduces a task, '{}' wraps parameters or inline context, '<>' defines the expected output "
    "schema, '->' indicates sequential steps, '|' separates alternatives, '@' marks multi-turn conversations, and "
    "'*n' denotes repetition.\n\n"
)


def generate_promptscript(task_name: str, input_text: str) -> str:
    """Return a PromptScript-formatted instruction with the standard explanation prefix."""
    # Escape curly braces within user input to avoid breaking the template
    safe_input = input_text.replace("{", "{{").replace("}", "}}")
    script_lines = [
        f":{task_name} {{input=\"{safe_input}\"}} -> :Validate {{checks=\"clarity,specificity,context\"}} ",
        "-> :Rewrite {style=\"professional\", domain=\"general\"} <optimized_prompt>",
    ]
    return PROMPTSCRIPT_EXPLANATION + "\n".join(script_lines)

@app.post("/api/chat/process", response_model=ChatResponse)
async def chat_process(request: ChatRequest):
    """Process user prompt: return similar library prompts, optimized prompt, tweaked top match"""
    user_prompt = request.prompt.strip()
    max_results = max(1, min(request.max_results, 10))  # safety bounds

    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # 1. Find similar prompts
    search_resp = index_manager.search(query=user_prompt, limit=max_results)
    matches = search_resp.items

    optimized_prompt: str = ""
    tweaked_match: str = ""

    if index_manager.llm_connector:
        try:
            enh = await index_manager.llm_connector.enhance_prompt(user_prompt, "improve")
            if isinstance(enh, dict):
                optimized_prompt = enh.get("enhanced_content", "")
        except Exception:
            pass

        # Tweak top match for user use-case
        if matches:
            try:
                top_content = matches[0].content
                tweak = await index_manager.llm_connector.enhance_prompt(top_content, "improve")
                if isinstance(tweak, dict):
                    tweaked_match = tweak.get("enhanced_content", "")
            except Exception:
                pass

    # Fallbacks if LLM not available or returns empty
    if not optimized_prompt:
        optimized_prompt = user_prompt

    if matches and not tweaked_match:
        tweaked_match = matches[0].content

    # --- Format outputs based on requested template ---
    fmt = request.format.lower() if request.format else "promptscript"

    if fmt == "promptscript":
        optimized_prompt = generate_promptscript("ImprovePrompt", optimized_prompt)
        tweaked_match = generate_promptscript("ImprovePrompt", tweaked_match) if tweaked_match else ""
    elif fmt == "yaml":
        optimized_prompt = generate_prompt_yaml("ImprovePrompt", optimized_prompt)
        tweaked_match = generate_prompt_yaml("ImprovePrompt", tweaked_match) if tweaked_match else ""
    else:  # plain
        # keep as-is (already plain improved)
        pass

    return ChatResponse(
        matches=matches,
        optimized_prompt=optimized_prompt,
        tweaked_match=tweaked_match,
    )

# ---------------------- LLM API Key Endpoint -----------------------------

@app.post("/api/llm/set-key")
async def set_llm_api_key(request: ApiKeyRequest):
    """Set or update the OpenRouter API key at runtime"""
    new_key = request.api_key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="API key is required")

    # Re-create connector with new key
    connector = create_llm_connector(new_key)
    if connector is None:
        raise HTTPException(status_code=500, detail="Failed to initialise LLM connector with provided key")

    index_manager.llm_connector = connector
    return {"status": "success", "message": "API key updated and LLM connector initialised"}

# ---------------------- Collections (Garden Beds) ------------------------

class CollectionCreate(BaseModel):
    name: str

class AddItemRequest(BaseModel):
    prompt_id: str

class CollectionDetail(BaseModel):
    id: int
    name: str
    items: List[str]

@app.get("/api/collections", response_model=List[Dict[str, Any]])
async def get_collections(user_id: str = Header(..., alias="X-User-Id")):
    """Return all collections for the given user"""
    return list_collections(user_id)

@app.post("/api/collections", status_code=201)
async def create_collection_endpoint(
    payload: CollectionCreate,
    user_id: str = Header(..., alias="X-User-Id"),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name required")
    col_id = create_collection(user_id, name)
    return {"id": col_id, "name": name}

@app.get("/api/collections/{collection_id}")
async def get_collection_endpoint(collection_id: int, user_id: str = Header(..., alias="X-User-Id")):
    # Verify collection belongs to user
    cols = list_collections(user_id)
    if not any(c["id"] == collection_id for c in cols):
        raise HTTPException(status_code=404, detail="Collection not found")
    item_ids = get_collection_items(collection_id)
    id_map = {it.id: it for it in index_manager.index}
    items_full = [id_map[pid] for pid in item_ids if pid in id_map]
    name = next(c["name"] for c in cols if c["id"] == collection_id)
    return {"id": collection_id, "name": name, "items": [it.dict() for it in items_full]}

@app.post("/api/collections/{collection_id}/items", status_code=204)
async def add_item_endpoint(
    collection_id: int,
    payload: AddItemRequest,
    user_id: str = Header(..., alias="X-User-Id"),
):
    cols = list_collections(user_id)
    if not any(c["id"] == collection_id for c in cols):
        raise HTTPException(status_code=404, detail="Collection not found")
    db_add_item(collection_id, payload.prompt_id)
    return {"status": "added"}

@app.delete("/api/collections/{collection_id}/items/{prompt_id}", status_code=204)
async def remove_item_endpoint(
    collection_id: int,
    prompt_id: str,
    user_id: str = Header(..., alias="X-User-Id"),
):
    cols = list_collections(user_id)
    if not any(c["id"] == collection_id for c in cols):
        raise HTTPException(status_code=404, detail="Collection not found")
    db_remove_item(collection_id, prompt_id)
    return {"status": "removed"}

@app.get("/api/guide/{item_id}")
async def get_prompt_guide(item_id: str):
    """Return contextual usage guide for a prompt using RAG + LLM enhancement."""
    # Find prompt
    matches = [it for it in index_manager.index if it.id == item_id]
    if not matches:
        raise HTTPException(status_code=404, detail="Prompt not found")
    prompt_item = matches[0]
    content = prompt_item.content

    # Placeholder cache (in-memory dict on index_manager)
    if not hasattr(index_manager, "_guide_cache"):
        index_manager._guide_cache = {}
    if item_id in index_manager._guide_cache:
        return index_manager._guide_cache[item_id]

    # RAG: Find related guides and articles for context
    related_guides = []
    try:
        # Search for related guides/articles using category and keywords
        category_guides = index_manager.search(query=f"{prompt_item.category} guide tips", category="", limit=3)
        if category_guides.items:
            related_guides.extend([g for g in category_guides.items if "guide" in g.title.lower() or "tips" in g.title.lower()])
        
        # Search by prompt title keywords
        title_words = prompt_item.title.split()[:3]  # First 3 words
        for word in title_words:
            word_guides = index_manager.search(query=f"{word} how to use", category="", limit=2)
            if word_guides.items:
                related_guides.extend([g for g in word_guides.items if g.id not in [rg.id for rg in related_guides]])
    except Exception:
        pass

    # Generate enhanced guide with RAG context
    summary = ""
    advice = ""
    tips = ""
    
    if index_manager.llm_connector:
        try:
            # Create context from related guides
            rag_context = "\n\n".join([f"Related Guide: {g.title}\n{g.description[:300]}" for g in related_guides[:3]])
            
            # Enhanced prompt with RAG context
            enhanced_guide_prompt = f"""
Create a practical usage guide for this AI prompt. Use the related guides as context for best practices.

TARGET PROMPT:
Title: {prompt_item.title}
Category: {prompt_item.category}
Content: {content[:500]}...

RELATED GUIDANCE CONTEXT:
{rag_context}

Please provide:
1. A concise summary (1-2 sentences)
2. Key usage tips (3-4 bullet points)
3. Best practices specific to this type of prompt

Focus on actionable advice that helps users get the best results."""

            # Generate comprehensive guide
            guide_result = await index_manager.llm_connector._make_request([
                {"role": "system", "content": "You are an expert prompt engineer. Provide practical, actionable guidance."},
                {"role": "user", "content": enhanced_guide_prompt}
            ], model="llama-3.1-8b")
            
            if guide_result:
                # Parse the response into sections
                lines = guide_result.split('\n')
                current_section = "summary"
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if "summary" in line.lower() or "overview" in line.lower():
                        current_section = "summary"
                    elif "tips" in line.lower() or "usage" in line.lower():
                        current_section = "tips"
                    elif "best practices" in line.lower() or "advice" in line.lower():
                        current_section = "advice"
                    elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                        if current_section == "tips":
                            tips += line + "\n"
                        else:
                            advice += line + "\n"
                    elif current_section == "summary" and len(line) > 20:
                        summary += line + " "
                
                # Fallback: use first part as summary if parsing failed
                if not summary and guide_result:
                    summary = guide_result.split('\n')[0][:200]
            
            # Generate additional use cases if no tips were extracted
            if not tips and not advice:
                use_cases = await index_manager.llm_connector.suggest_use_cases(content)
                advice = "\n".join(f"• {uc}" for uc in (use_cases or []))
                
        except Exception as e:
            print(f"Error generating enhanced guide: {e}")
            # Fallback to basic generation
            try:
                summary = await index_manager.llm_connector.summarize_prompt(content, 200) or ""
                use_cases = await index_manager.llm_connector.suggest_use_cases(content)
                advice = "\n".join(f"• {uc}" for uc in (use_cases or []))
            except Exception:
                pass
    
    # Final fallback
    if not summary:
        summary = f"A {prompt_item.category.lower()} prompt for {prompt_item.title.lower()}."
    
    guide = {
        "summary": summary.strip(),
        "advice": advice.strip(),
        "tips": tips.strip(),
        "related_guides_count": len(related_guides)
    }
    
    index_manager._guide_cache[item_id] = guide
    return guide

@app.get("/api/popular")
async def get_popular_prompts(limit: int = Query(10)):
    """Return most viewed prompts"""
    pops = get_popular(limit)
    # join with prompt details
    id_map = {it.id: it for it in index_manager.index}
    items = []
    for p in pops:
        item = id_map.get(p["prompt_id"])
        if item:
            items.append({"prompt": item.dict(), **p})
    return items

@app.get("/api/trending")
async def get_trending_prompts(limit: int = Query(10)):
    """Return trending prompts (hot in last 7 days)"""
    trending = get_trending(limit)
    id_map = {it.id: it for it in index_manager.index}
    items = []
    for t in trending:
        item = id_map.get(t["prompt_id"])
        if item:
            items.append({"prompt": item.dict(), **t})
    return items

@app.get("/api/most-grafted")
async def get_most_grafted_prompts(limit: int = Query(10)):
    """Return most grafted (remixed) prompts"""
    grafted = get_most_grafted(limit)
    id_map = {it.id: it for it in index_manager.index}
    items = []
    for g in grafted:
        item = id_map.get(g["prompt_id"])
        if item:
            items.append({"prompt": item.dict(), **g})
    return items

@app.get("/api/new-sprouts")
async def get_new_sprouts_prompts(limit: int = Query(10)):
    """Return recently discovered prompts (new sprouts)"""
    sprouts = get_new_sprouts(limit)
    id_map = {it.id: it for it in index_manager.index}
    items = []
    for s in sprouts:
        item = id_map.get(s["prompt_id"])
        if item:
            items.append({"prompt": item.dict(), **s})
    return items

@app.get("/api/discovery-signals")
async def get_discovery_signals():
    """Return all discovery signals for the homepage"""
    id_map = {it.id: it for it in index_manager.index}
    
    def enrich_items(items_data):
        enriched = []
        for item_data in items_data:
            item = id_map.get(item_data["prompt_id"])
            if item:
                enriched.append({"prompt": item.dict(), **item_data})
        return enriched
    
    return {
        "trending_blossoms": enrich_items(get_trending(6)),
        "most_grafted": enrich_items(get_most_grafted(6)),  
        "new_sprouts": enrich_items(get_new_sprouts(6)),
        "popular_classics": enrich_items(get_popular(6))
    }

# ---------------------- Collection Management ----------------------------

@app.put("/api/collections/{collection_id}")
async def rename_collection_endpoint(
    collection_id: int,
    payload: CollectionCreate,
    user_id: str = Header(..., alias="X-User-Id"),
):
    """Rename an existing collection (garden bed)"""
    cols = list_collections(user_id)
    if not any(c["id"] == collection_id for c in cols):
        raise HTTPException(status_code=404, detail="Collection not found")
    new_name = payload.name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name required")
    db_rename_collection(collection_id, new_name)
    return {"id": collection_id, "name": new_name}


@app.delete("/api/collections/{collection_id}")
async def delete_collection_endpoint(
    collection_id: int,
    user_id: str = Header(..., alias="X-User-Id"),
):
    """Delete a collection and its items"""
    cols = list_collections(user_id)
    if not any(c["id"] == collection_id for c in cols):
        raise HTTPException(status_code=404, detail="Collection not found")
    db_delete_collection(collection_id)
    return {"status": "deleted"}


# ---------------------- Prompt Chains (Vines) Management -----------------

class ChainCreate(BaseModel):
    name: str
    description: str = ""
    prompts: List[Dict[str, Any]]

class ChainUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    prompts: Optional[List[Dict[str, Any]]] = None

@app.get("/api/chains")
async def list_user_chains(user_id: str = Header(..., alias="X-User-Id")):
    """List all prompt chains for a user"""
    chains = list_chains(user_id)
    # Parse prompts JSON for each chain
    for chain in chains:
        try:
            import json
            chain['prompts'] = json.loads(chain['prompts'])
        except:
            chain['prompts'] = []
    return chains

@app.post("/api/chains")
async def create_new_chain(
    chain: ChainCreate,
    user_id: str = Header(..., alias="X-User-Id")
):
    """Create a new prompt chain"""
    import json
    prompts_json = json.dumps(chain.prompts)
    chain_id = create_chain(user_id, chain.name, chain.description, prompts_json)
    return {"id": chain_id, "name": chain.name, "description": chain.description}

@app.get("/api/chains/{chain_id}")
async def get_chain_details(
    chain_id: int,
    user_id: str = Header(..., alias="X-User-Id")
):
    """Get details of a specific chain"""
    chain = get_chain(chain_id, user_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    # Parse prompts JSON
    try:
        import json
        chain['prompts'] = json.loads(chain['prompts'])
    except:
        chain['prompts'] = []
    
    return chain

@app.put("/api/chains/{chain_id}")
async def update_chain_endpoint(
    chain_id: int,
    updates: ChainUpdate,
    user_id: str = Header(..., alias="X-User-Id")
):
    """Update a prompt chain"""
    chain = get_chain(chain_id, user_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    import json
    prompts_json = json.dumps(updates.prompts) if updates.prompts is not None else None
    
    update_chain(
        chain_id, 
        user_id, 
        name=updates.name,
        description=updates.description,
        prompts=prompts_json
    )
    
    return {"status": "updated"}

@app.delete("/api/chains/{chain_id}")
async def delete_chain_endpoint(
    chain_id: int,
    user_id: str = Header(..., alias="X-User-Id")
):
    """Delete a prompt chain"""
    chain = get_chain(chain_id, user_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    delete_chain(chain_id, user_id)
    return {"status": "deleted"}

@app.post("/api/chains/{chain_id}/execute")
async def execute_chain(
    chain_id: int,
    user_id: str = Header(..., alias="X-User-Id")
):
    """Execute a prompt chain sequentially"""
    if not index_manager.llm_connector:
        raise HTTPException(status_code=501, detail="LLM connector not available")
    
    chain = get_chain(chain_id, user_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    import json
    try:
        prompts = json.loads(chain['prompts'])
    except:
        raise HTTPException(status_code=400, detail="Invalid chain prompts")
    
    if not prompts:
        raise HTTPException(status_code=400, detail="Chain has no prompts")
    
    results = []
    previous_output = None
    
    for i, prompt_data in enumerate(prompts):
        try:
            prompt_content = prompt_data.get('content', '')
            prompt_title = prompt_data.get('title', f'Step {i+1}')
            
            # If this isn't the first prompt, use previous output as input
            if previous_output and '{previous_output}' in prompt_content:
                prompt_content = prompt_content.replace('{previous_output}', previous_output)
            
            # Execute the prompt
            messages = [
                {"role": "user", "content": prompt_content}
            ]
            
            output = await index_manager.llm_connector._make_request(messages)
            
            if output:
                results.append({
                    "step": i + 1,
                    "title": prompt_title,
                    "prompt": prompt_content,
                    "output": output,
                    "success": True
                })
                previous_output = output
            else:
                results.append({
                    "step": i + 1,
                    "title": prompt_title,
                    "prompt": prompt_content,
                    "output": "No response from LLM",
                    "success": False
                })
                break
                
        except Exception as e:
            results.append({
                "step": i + 1,
                "title": prompt_data.get('title', f'Step {i+1}'),
                "prompt": prompt_data.get('content', ''),
                "output": f"Error: {str(e)}",
                "success": False
            })
            break
    
    return {
        "chain_id": chain_id,
        "chain_name": chain['name'],
        "results": results,
        "final_output": results[-1]['output'] if results else None
    }

# ---------------------- Prompt Quality Grading ---------------------------

class PromptGradeRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-4o"
    use_llm: Optional[bool] = True

@app.post("/api/prompt/grade")
async def grade_prompt_endpoint(payload: PromptGradeRequest):
    """Grade a prompt's quality using heuristics and, if configured, an LLM self-evaluation."""
    from prompt_quality_grader import PromptGrader  # local import to avoid circular deps

    grader = PromptGrader(index_manager.llm_connector)
    result = grader.grade(
        payload.prompt,
        model=payload.model or "gpt-4o",
        use_llm=payload.use_llm,
    )
    return result

# ---------------------- AI News RSS ---------------------------

@app.get("/api/ai-news")
async def get_ai_news(limit: int = Query(20, description="Max news items")):
    """Return latest AI-related headlines aggregated from RSS feeds."""
    items = rss_fetcher.get_ai_news()[:limit]
    return {"items": items}

# -------------------- Prompt Feedback & Iteration --------------------

class PromptFeedbackRequest(BaseModel):
    feedback: str
    enhancement_type: Optional[str] = "improve"

class PromptVersionResponse(BaseModel):
    version_id: int
    version_number: int
    improved_content: str
    created_at: str


@app.post("/api/prompts/{prompt_id}/feedback", response_model=PromptVersionResponse)
async def iterate_prompt(
    prompt_id: str,
    payload: PromptFeedbackRequest,
    user_id: str = Header(..., alias="X-User-Id"),
):
    """Generate an improved version of a prompt based on user feedback and store it."""
    # Locate the original prompt in the current index
    original_items = [item for item in index_manager.index if item.id == prompt_id]
    if not original_items:
        raise HTTPException(status_code=404, detail="Prompt not found")
    original_item = original_items[0]

    # Ensure LLM connector is available
    if not index_manager.llm_connector:
        raise HTTPException(status_code=503, detail="LLM connector not available")

    # Combine original prompt with explicit feedback for the model
    combined_prompt = original_item.content
    if payload.feedback:
        combined_prompt = f"{combined_prompt}\n\nUser feedback to address:\n{payload.feedback.strip()}"

    try:
        enhanced = await index_manager.llm_connector.enhance_prompt(
            combined_prompt, enhancement_type=payload.enhancement_type or "improve"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {e}")

    if not enhanced or not enhanced.get("enhanced_content"):
        raise HTTPException(status_code=500, detail="Failed to generate improved prompt")

    # Persist the new version in the DB
    version_row_id = create_prompt_version(
        original_prompt_id=prompt_id,
        user_id=user_id,
        feedback=payload.feedback,
        enhancement_type=payload.enhancement_type or "improve",
        improved_content=enhanced["enhanced_content"],
    )

    # Retrieve version_number for response
    versions = list_prompt_versions(prompt_id)
    created_version = next((v for v in versions if v["id"] == version_row_id), None)
    if not created_version:
        created_version = {
            "id": version_row_id,
            "version_number": len(versions),
            "improved_content": enhanced["enhanced_content"],
            "created_at": None,
        }

    return PromptVersionResponse(
        version_id=created_version["id"],
        version_number=created_version["version_number"],
        improved_content=created_version["improved_content"],
        created_at=created_version.get("created_at", ""),
    )

@app.get("/api/prompts/{prompt_id}/versions")
async def get_prompt_versions(prompt_id: str):
    """Return all stored versions of a prompt"""
    return list_prompt_versions(prompt_id)

# ---------------------- Alternative Prompt Generators ----------------------------

def generate_prompt_yaml(task_name: str, input_text: str) -> str:
    """Return a YAML-formatted instruction block covering the same workflow used in PromptScript."""
    # Indent the user input safely for YAML literal block style
    indented_input = "\n".join(["  " + line for line in input_text.split("\n")])
    yaml_content = (
        f"# {task_name} prompt generated in YAML format\n"
        "task: ImprovePrompt\n"
        "input: |\n"
        f"{indented_input}\n"
        "steps:\n"
        "  - Validate: [clarity, specificity, context]\n"
        "  - Rewrite:\n"
        "      style: professional\n"
        "      domain: general\n"
        "output: optimized_prompt\n"
    )
    return yaml_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)