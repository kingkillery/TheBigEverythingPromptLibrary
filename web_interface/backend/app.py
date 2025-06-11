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

# Add collections_db helper
from collections_db import (
    create_collection,
    list_collections,
    add_item as db_add_item,
    remove_item as db_remove_item,
    get_collection_items,
)

app = FastAPI(title="Prompt Library API", version="1.0.0")

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
async def read_root(request: Request):
    """Serve the frontend HTML"""
    return templates.TemplateResponse("index.html", {"request": request})

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
    sort_by: str = Query("relevance", description="Sort by: relevance, quality, title, newest")
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
        sort_by=sort_by
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

    # --- Wrap outputs in PromptScript framework ---
    optimized_prompt = generate_promptscript("ImprovePrompt", optimized_prompt)
    tweaked_match = generate_promptscript("ImprovePrompt", tweaked_match) if tweaked_match else ""

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

@app.get("/api/collections/{collection_id}", response_model=CollectionDetail)
async def get_collection_endpoint(collection_id: int, user_id: str = Header(..., alias="X-User-Id")):
    # Verify collection belongs to user
    cols = list_collections(user_id)
    if not any(c["id"] == collection_id for c in cols):
        raise HTTPException(status_code=404, detail="Collection not found")
    items = get_collection_items(collection_id)
    name = next(c["name"] for c in cols if c["id"] == collection_id)
    return CollectionDetail(id=collection_id, name=name, items=items)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)