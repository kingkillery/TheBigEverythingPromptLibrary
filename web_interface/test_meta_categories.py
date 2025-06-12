#!/usr/bin/env python3
"""
Test script to verify meta-categories are working in the index
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent.parent / ".scripts"))

# Create a minimal index manager to test categorization
from category_config import suggest_category, CATEGORIES, get_all_categories
import gptparser

class MockPromptItem:
    def __init__(self, id, title, description, content, category="CustomInstructions", 
                 subcategory="ChatGPT", meta_category=None, meta_subcategory=None):
        self.id = id
        self.title = title
        self.description = description
        self.content = content
        self.category = category
        self.subcategory = subcategory
        self.meta_category = meta_category
        self.meta_subcategory = meta_subcategory
        self.url = ""
        self.file_path = ""
        self.tags = []
        self.created_date = None
        self.version = ""

def test_meta_categorization():
    """Test that GPT files are getting properly categorized"""
    print("=== Testing Meta-Categorization ===\n")
    
    # Test category configuration
    print("1. Category Configuration:")
    all_cats = get_all_categories()
    print(f"   Total categories: {len(all_cats)}")
    for cat in all_cats[:3]:  # Show first 3
        print(f"   - {cat['name']}: {len(cat['subcategories'])} subcategories")
    print()
    
    # Test GPT processing with categorization
    print("2. Processing GPT Files with Meta-Categories:")
    items = []
    count = 0
    
    for ok, gpt in gptparser.enum_gpts():
        if not ok or count >= 20:  # Test first 20
            continue
        count += 1
        
        gpt_id = gpt.id()
        if not gpt_id:
            continue
            
        title = gpt.get('title', '') or ''
        description = gpt.get('description', '') or ''
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
        
        item = MockPromptItem(
            id=gpt_id.id,
            title=title,
            description=description,
            content=instructions,
            meta_category=meta_category,
            meta_subcategory=meta_subcategory
        )
        items.append(item)
        
        print(f"   {count:2d}. {title[:40]:<40} → {meta_category or 'None'}")
    
    print(f"\n   Processed {len(items)} items")
    
    # Test meta-category counts
    print("\n3. Meta-Category Distribution:")
    meta_counts = {}
    
    for item in items:
        if item.meta_category:
            if item.meta_category not in meta_counts:
                meta_counts[item.meta_category] = {"count": 0, "name": ""}
            meta_counts[item.meta_category]["count"] += 1
            
            # Get category name
            if item.meta_category in CATEGORIES:
                meta_counts[item.meta_category]["name"] = CATEGORIES[item.meta_category].name
    
    for cat_id, data in meta_counts.items():
        print(f"   {data['name']}: {data['count']} items")
    
    uncategorized = len([item for item in items if not item.meta_category])
    print(f"   Uncategorized: {uncategorized} items")
    
    print(f"\n=== Test Complete ===")
    print(f"Meta-categorization success rate: {(len(items) - uncategorized) / len(items) * 100:.1f}%")

if __name__ == "__main__":
    test_meta_categorization()