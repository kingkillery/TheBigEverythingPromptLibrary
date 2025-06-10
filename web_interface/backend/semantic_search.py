"""
Optional semantic search enhancement using sentence transformers
Install with: pip install sentence-transformers numpy
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    print("Semantic search not available. Install with: pip install sentence-transformers numpy")

class SemanticSearchEngine:
    """
    Semantic search engine using sentence transformers for meaning-based search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        if not SEMANTIC_SEARCH_AVAILABLE:
            raise ImportError("sentence-transformers and numpy required for semantic search")
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "embeddings_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load the model (lightweight and fast)
        print(f"Loading semantic search model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self.embeddings = None
        self.items = None
        self.embedding_cache_file = self.cache_dir / f"embeddings_{model_name.replace('/', '_')}.npz"
        self.metadata_cache_file = self.cache_dir / f"metadata_{model_name.replace('/', '_')}.json"
    
    def _prepare_text(self, item) -> str:
        """Prepare text for embedding"""
        # Combine title, description, and key content for embedding
        text_parts = []
        
        if item.title:
            text_parts.append(f"Title: {item.title}")
        
        if item.description:
            text_parts.append(f"Description: {item.description}")
        
        if item.tags:
            text_parts.append(f"Tags: {', '.join(item.tags)}")
        
        # Add truncated content
        content = item.content[:500] if item.content else ""
        if content:
            text_parts.append(f"Content: {content}")
        
        return " ".join(text_parts)
    
    def build_embeddings(self, items: List) -> bool:
        """Build embeddings for all items"""
        try:
            print(f"Building semantic embeddings for {len(items)} items...")
            
            # Prepare texts for embedding
            texts = [self._prepare_text(item) for item in items]
            item_ids = [item.id for item in items]
            
            # Check if we have cached embeddings
            if self._load_cached_embeddings(item_ids):
                print("Loaded cached embeddings")
                return True
            
            # Generate embeddings
            print("Generating new embeddings...")
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            
            # Cache embeddings
            self._save_embeddings(embeddings, item_ids)
            
            self.embeddings = embeddings
            self.items = items
            
            print(f"✅ Semantic search ready with {len(items)} items")
            return True
            
        except Exception as e:
            print(f"❌ Error building embeddings: {e}")
            return False
    
    def _load_cached_embeddings(self, current_item_ids: List[str]) -> bool:
        """Load cached embeddings if they match current items"""
        try:
            if not self.embedding_cache_file.exists() or not self.metadata_cache_file.exists():
                return False
            
            # Load metadata
            with open(self.metadata_cache_file, 'r') as f:
                metadata = json.load(f)
            
            cached_ids = metadata.get('item_ids', [])
            
            # Check if cached IDs match current IDs
            if set(cached_ids) != set(current_item_ids):
                print("Cache outdated - item IDs don't match")
                return False
            
            # Load embeddings
            data = np.load(self.embedding_cache_file)
            self.embeddings = data['embeddings']
            
            # Reorder items to match cached order
            id_to_item = {item.id: item for item in self.items if hasattr(self, 'items') and self.items}
            if not id_to_item:
                return False
                
            self.items = [id_to_item[item_id] for item_id in cached_ids if item_id in id_to_item]
            
            return len(self.items) == len(cached_ids)
            
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
            return False
    
    def _save_embeddings(self, embeddings: np.ndarray, item_ids: List[str]):
        """Save embeddings and metadata to cache"""
        try:
            # Save embeddings
            np.savez_compressed(self.embedding_cache_file, embeddings=embeddings)
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'item_ids': item_ids,
                'embedding_shape': embeddings.shape,
                'created_at': str(np.datetime64('now'))
            }
            
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving embeddings cache: {e}")
    
    def semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[float, any]]:
        """
        Perform semantic search
        Returns list of (similarity_score, item) tuples
        """
        if self.embeddings is None or self.items is None:
            return []
        
        try:
            # Encode the query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Minimum similarity threshold
                    results.append((float(similarities[idx]), self.items[idx]))
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def hybrid_search(self, query: str, keyword_results: List, semantic_weight: float = 0.3) -> List:
        """
        Combine keyword and semantic search results
        """
        if not self.embeddings is not None:
            return keyword_results
        
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=100)
        
        # Create a mapping of semantic scores
        semantic_scores = {item.id: score for score, item in semantic_results}
        
        # Combine scores
        hybrid_results = []
        for item in keyword_results:
            keyword_score = 1.0  # Base score for keyword matches
            semantic_score = semantic_scores.get(item.id, 0.0)
            
            # Weighted combination
            combined_score = (1 - semantic_weight) * keyword_score + semantic_weight * semantic_score
            hybrid_results.append((combined_score, item))
        
        # Add purely semantic results that weren't in keyword results
        keyword_ids = {item.id for item in keyword_results}
        for score, item in semantic_results:
            if item.id not in keyword_ids:
                combined_score = semantic_weight * score
                hybrid_results.append((combined_score, item))
        
        # Sort by combined score
        hybrid_results.sort(key=lambda x: x[0], reverse=True)
        
        return [item for score, item in hybrid_results]
    
    def find_similar(self, item_id: str, top_k: int = 10) -> List[Tuple[float, any]]:
        """Find items similar to a given item"""
        if self.embeddings is None or self.items is None:
            return []
        
        try:
            # Find the item
            item_idx = None
            for i, item in enumerate(self.items):
                if item.id == item_id:
                    item_idx = i
                    break
            
            if item_idx is None:
                return []
            
            # Get the item's embedding
            item_embedding = self.embeddings[item_idx:item_idx+1]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, item_embedding.T).flatten()
            
            # Get top results (excluding the item itself)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.4:  # Minimum similarity threshold
                    results.append((float(similarities[idx]), self.items[idx]))
            
            return results
            
        except Exception as e:
            print(f"Error finding similar items: {e}")
            return []

# Factory function for optional semantic search
def create_semantic_search_engine() -> Optional[SemanticSearchEngine]:
    """Create semantic search engine if dependencies are available"""
    if not SEMANTIC_SEARCH_AVAILABLE:
        return None
    
    try:
        return SemanticSearchEngine()
    except Exception as e:
        print(f"Could not initialize semantic search: {e}")
        return None