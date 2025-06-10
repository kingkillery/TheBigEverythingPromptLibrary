#!/usr/bin/env python3
"""
Test script for enhanced search functionality
Run this to validate the improved search capabilities
"""

import sys
import asyncio
from pathlib import Path

# Add the backend directory to path
sys.path.append(str(Path(__file__).parent / "backend"))

from app import IndexManager

def test_enhanced_search():
    """Test the enhanced search functionality"""
    print("🔍 Testing Enhanced Search Functionality")
    print("=" * 50)
    
    # Initialize the index manager
    print("📚 Building search index...")
    index_manager = IndexManager()
    
    print(f"✅ Indexed {len(index_manager.index)} items")
    
    # Test basic search
    print("\n🔎 Testing basic search...")
    results = index_manager.search(query="coding assistant")
    print(f"Found {results.total} results for 'coding assistant'")
    
    if results.items:
        top_result = results.items[0]
        print(f"Top result: {top_result.title}")
        print(f"Category: {top_result.category}")
        print(f"Tags: {', '.join(top_result.tags)}")
    
    # Test quality filtering
    print("\n⭐ Testing quality filtering...")
    high_quality_results = index_manager.search(query="", min_quality=0.7, limit=10)
    print(f"Found {high_quality_results.total} high-quality items (0.7+ score)")
    
    if high_quality_results.items:
        for i, item in enumerate(high_quality_results.items[:3], 1):
            print(f"{i}. {item.title} ({item.category})")
    
    # Test sorting options
    print("\n📊 Testing sorting options...")
    quality_sorted = index_manager.search(query="", sort_by="quality", limit=5)
    print(f"Top 5 by quality:")
    for i, item in enumerate(quality_sorted.items, 1):
        print(f"{i}. {item.title}")
    
    # Test category filtering
    print("\n📂 Testing category filtering...")
    guide_results = index_manager.search(category="Guides")
    print(f"Found {guide_results.total} items in Guides category")
    
    # Test enhanced search features if available
    if index_manager.enhanced_search:
        print("\n🚀 Enhanced search is available!")
        
        # Test fuzzy search
        print("\nTesting fuzzy search...")
        fuzzy_results = index_manager.search(query="securty analisis")  # Intentional typos
        print(f"Fuzzy search found {fuzzy_results.total} results for misspelled query")
        
        # Test quality distribution
        print("\nTesting quality analysis...")
        if fuzzy_results.filters and 'search_stats' in fuzzy_results.filters:
            stats = fuzzy_results.filters['search_stats']
            print(f"Average quality: {stats['avg_quality']:.2f}")
            print(f"Quality distribution: {stats['quality_distribution']}")
    else:
        print("\n⚠️  Enhanced search not available (missing dependencies)")
    
    # Test semantic search if available
    if index_manager.semantic_search:
        print("\n🧠 Semantic search is available!")
        semantic_results = index_manager.search(query="natural language processing machine learning")
        print(f"Semantic search found {semantic_results.total} conceptually related results")
    else:
        print("\n💡 Semantic search not available (install: pip install sentence-transformers)")
    
    # Test LLM integration if available
    if index_manager.llm_connector:
        print("\n🤖 LLM integration is available!")
        print("AI-powered prompt enhancement and analysis ready")
    else:
        print("\n🔌 LLM integration not available")
    
    print("\n" + "=" * 50)
    print("✅ Enhanced search test completed!")
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"   Total items indexed: {len(index_manager.index)}")
    print(f"   Categories available: {len(index_manager.get_categories())}")
    print(f"   Tags available: {len(index_manager.get_tags())}")
    print(f"   Enhanced search: {'✅' if index_manager.enhanced_search else '❌'}")
    print(f"   Semantic search: {'✅' if index_manager.semantic_search else '❌'}")
    print(f"   LLM integration: {'✅' if index_manager.llm_connector else '❌'}")

async def test_api_endpoints():
    """Test the API endpoints"""
    print("\n🌐 Testing API endpoints...")
    
    try:
        import httpx
        
        # Test if server is running
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8000/api/categories", timeout=5)
                if response.status_code == 200:
                    categories = response.json()
                    print(f"✅ API is running - {len(categories)} categories available")
                    return True
                else:
                    print(f"❌ API returned status {response.status_code}")
                    return False
            except httpx.ConnectError:
                print("❌ API server not running. Start with: python start_server.py")
                return False
    except ImportError:
        print("❌ httpx not available for API testing")
        return False

def main():
    """Main test function"""
    import sys
    
    # Check if quick mode (for Docker)
    quick_mode = '--quick' in sys.argv
    
    print("🧪 Testing Enhanced Prompt Library Search")
    print("=" * 60)
    
    # Test the core search functionality
    test_enhanced_search()
    
    # Test API if possible (skip in quick mode)
    if not quick_mode:
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(test_api_endpoints())
        except Exception as e:
            print(f"API test skipped: {e}")
    
    print("\n🎉 All tests completed!")
    if not quick_mode:
        print("\nTo start the web interface:")
        print("   cd web_interface")
        print("   python start_server.py")
        print("   Open: http://localhost:8000")

if __name__ == "__main__":
    main()