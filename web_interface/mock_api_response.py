#!/usr/bin/env python3
"""
Mock API response to show what the frontend should receive
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent.parent / ".scripts"))

from category_config import CATEGORIES, get_all_categories
import gptparser

def generate_mock_responses():
    """Generate mock API responses to show what frontend should receive"""
    
    print("🌐 Mock API Responses for Frontend")
    print("=" * 60)
    
    # 1. /api/meta-categories endpoint
    print("\n📡 GET /api/meta-categories")
    print("-" * 40)
    
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
    
    # Count some sample items
    from category_config import suggest_category
    sample_count = 0
    for ok, gpt in gptparser.enum_gpts():
        if not ok or sample_count >= 100:
            continue
        sample_count += 1
        
        title = gpt.get('title', '') or ''
        description = gpt.get('description', '') or ''
        instructions = gpt.get('instructions', '') or ''
        content_text = f"{title} {description} {instructions}"
        
        suggested_cat = suggest_category(content_text)
        if suggested_cat:
            cat_id, subcat_id = suggested_cat
            if cat_id in meta_counts:
                meta_counts[cat_id]["count"] += 1
                if subcat_id in meta_counts[cat_id]["subcategories"]:
                    meta_counts[cat_id]["subcategories"][subcat_id]["count"] += 1
    
    print(json.dumps(meta_counts, indent=2)[:1000] + "...")
    
    # 2. /api/category-config endpoint
    print("\n\n📡 GET /api/category-config")
    print("-" * 40)
    
    config = get_all_categories()
    print(json.dumps(config[:2], indent=2))  # Show first 2 categories
    print("...")
    
    # 3. Sample search response with meta-categories
    print("\n\n📡 GET /api/search?meta_category=coding_development")
    print("-" * 40)
    
    sample_items = []
    count = 0
    for ok, gpt in gptparser.enum_gpts():
        if not ok or count >= 5:
            continue
        
        title = gpt.get('title', '') or ''
        description = gpt.get('description', '') or ''
        instructions = gpt.get('instructions', '') or ''
        content_text = f"{title} {description} {instructions}"
        
        suggested_cat = suggest_category(content_text)
        if suggested_cat and suggested_cat[0] == "coding_development":
            count += 1
            
            sample_item = {
                "id": gpt.id().id if gpt.id() else f"gpt_{count}",
                "title": title,
                "description": description[:100] + "...",
                "category": "CustomInstructions",
                "subcategory": "ChatGPT",
                "meta_category": suggested_cat[0],
                "meta_subcategory": suggested_cat[1],
                "tags": ["coding", "development"],
                "content": instructions[:200] + "..."
            }
            sample_items.append(sample_item)
    
    search_response = {
        "items": sample_items,
        "total": len(sample_items),
        "query": "",
        "filters": {
            "meta_category": "coding_development",
            "meta_subcategory": ""
        }
    }
    
    print(json.dumps(search_response, indent=2))
    
    print("\n\n📋 FRONTEND INTEGRATION NOTES")
    print("=" * 60)
    print("✅ The categorization system provides:")
    print("   • Rich category metadata with colors and icons")
    print("   • Hierarchical subcategories")
    print("   • Item counts for each category/subcategory")
    print("   • Search filtering by meta-category and meta-subcategory")
    print()
    print("🎨 Frontend can use category.color for theming")
    print("🔍 Frontend can build category dropdowns from the hierarchy")
    print("📊 Frontend can show category counts in the UI")
    print("🏷️ Frontend can display category badges with icons")

if __name__ == "__main__":
    generate_mock_responses()