#!/usr/bin/env python3
"""
Test script for the new categorization system
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from category_config import (
    CATEGORIES,
    suggest_category,
    get_all_categories,
    get_category_hierarchy
)

def test_categorization():
    """Test the categorization system with some examples"""
    
    print("=== Testing Category Configuration ===\n")
    
    # Test 1: Show all categories
    print("1. All Categories:")
    for cat_id, category in CATEGORIES.items():
        print(f"   {category.name} ({cat_id})")
        print(f"      {category.description}")
        print(f"      Subcategories: {len(category.subcategories)}")
        print()
    
    # Test 2: Test auto-categorization
    print("\n2. Auto-categorization Tests:")
    
    test_cases = [
        ("Code review assistant that helps debug Python applications", "coding_development"),
        ("Generate creative writing prompts for fiction stories", "writing_content"),
        ("Penetration testing helper for ethical hackers", "security_hacking"),
        ("Dating advice and relationship counseling bot", "lifestyle_personal"),
        ("Interactive RPG adventure game with magic", "games_entertainment"),
        ("Math tutor for algebra and calculus students", "education_learning"),
        ("Business plan generator for startups", "business_professional"),
        ("Meditation guide with mindfulness exercises", "spiritual_philosophical"),
        ("AI image generator for digital art", "creative_art"),
        ("Prompt injection testing and security hardening", "meta_tools")
    ]
    
    for description, expected_category in test_cases:
        result = suggest_category(description)
        if result:
            cat_id, subcat_id = result
            category = CATEGORIES[cat_id]
            subcategory = next(s for s in category.subcategories if s.id == subcat_id)
            
            status = "✅" if cat_id == expected_category else "❌"
            print(f"   {status} '{description[:50]}...'")
            print(f"      → {category.name} > {subcategory.name}")
            if cat_id != expected_category:
                print(f"      Expected: {expected_category}")
        else:
            print(f"   ❌ '{description[:50]}...' → No category suggested")
        print()
    
    # Test 3: Category hierarchy
    print("\n3. Category Hierarchy:")
    hierarchy = get_category_hierarchy()
    for category, subcategories in hierarchy.items():
        print(f"   {category}:")
        for subcat in subcategories[:3]:  # Show first 3
            print(f"      - {subcat}")
        if len(subcategories) > 3:
            print(f"      ... and {len(subcategories) - 3} more")
        print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_categorization()