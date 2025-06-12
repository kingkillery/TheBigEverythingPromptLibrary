"""
Enhanced Search Engine with Quality Scoring and Advanced Filtering
"""

import re
import math
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import Counter
from fuzzywuzzy import fuzz
import numpy as np

class QualityScorer:
    """Scores prompts based on quality indicators"""
    
    def __init__(self):
        # Quality indicators
        self.positive_indicators = {
            'structure': ['step by step', 'systematic', 'methodical', 'structured', 'organized'],
            'specificity': ['specific', 'detailed', 'precise', 'exact', 'clear instructions'],
            'examples': ['example', 'for instance', 'such as', 'like', 'demonstrate'],
            'context': ['context', 'background', 'situation', 'scenario', 'environment'],
            'output_format': ['format', 'structure', 'template', 'layout', 'output'],
            'expertise': ['expert', 'professional', 'advanced', 'experienced', 'skilled'],
            'constraints': ['requirements', 'constraints', 'limitations', 'guidelines', 'rules']
        }
        
        self.negative_indicators = [
            'act as', 'pretend', 'role play', 'imagine you are',
            'hello', 'hi there', 'please help', 'can you',
            'simple', 'basic', 'easy', 'quick'
        ]
        
        # Technical depth indicators
        self.technical_indicators = [
            'algorithm', 'implementation', 'optimization', 'analysis', 'methodology',
            'framework', 'architecture', 'design pattern', 'best practices', 'workflow'
        ]
    
    def score_prompt_quality(self, item) -> float:
        """Score a prompt's quality from 0.0 to 1.0"""
        text = f"{item.title} {item.description} {item.content}".lower()
        score = 0.5  # Base score
        
        # Length bonus (longer prompts often have more detail)
        content_length = len(item.content)
        if content_length > 1000:
            score += 0.15
        elif content_length > 500:
            score += 0.1
        elif content_length < 100:
            score -= 0.15
        
        # Positive indicators
        for category, indicators in self.positive_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text)
            if matches > 0:
                score += min(matches * 0.05, 0.15)  # Cap bonus per category
        
        # Technical depth
        technical_matches = sum(1 for indicator in self.technical_indicators if indicator in text)
        if technical_matches > 0:
            score += min(technical_matches * 0.08, 0.2)
        
        # Negative indicators
        negative_matches = sum(1 for indicator in self.negative_indicators if indicator in text)
        if negative_matches > 0:
            score -= min(negative_matches * 0.1, 0.3)
        
        # Structure indicators (markdown headers, lists, etc.)
        if '#' in item.content:
            score += 0.05
        if any(marker in item.content for marker in ['1.', '2.', '-', '*']):
            score += 0.05
        if '```' in item.content:  # Code blocks
            score += 0.1
        
        # Title quality
        title_words = len(item.title.split())
        if 3 <= title_words <= 8:
            score += 0.05
        elif title_words < 2:
            score -= 0.1
        
        # Description quality
        if len(item.description) > 50 and len(item.description) < 500:
            score += 0.05
        
        # Tag diversity (more specific tags = higher quality)
        unique_tags = len(set(item.tags))
        if unique_tags >= 3:
            score += 0.1
        elif unique_tags >= 2:
            score += 0.05
        
        return max(0.0, min(1.0, score))

class EnhancedSearchEngine:
    """Enhanced search with fuzzy matching, quality scoring, and advanced filtering"""
    
    def __init__(self, config: Optional[dict] = None):
        self.quality_scorer = QualityScorer()
        # Default stop words; can be overridden via config
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could'
        }
        # Default scoring weights; can be overridden via config
        self.weights = {
            'quality': 0.25,
            'title': 0.30,
            'description': 0.15,
            'content': 0.10,
            'tags': 0.10,
            'semantic': 0.08,
            'category': 0.02
        }
        # Apply incoming config if provided
        if config:
            self.update_config(config)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Remove stop words and short words
        keywords = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Return most common keywords
        counter = Counter(keywords)
        return [word for word, count in counter.most_common(20)]
    
    def fuzzy_score(self, query: str, text: str) -> float:
        """Calculate fuzzy matching score"""
        if not query or not text:
            return 0.0
        
        # Direct substring match
        if query.lower() in text.lower():
            return 1.0
        
        # Fuzzy ratio
        fuzzy_ratio = fuzz.partial_ratio(query.lower(), text.lower()) / 100.0
        
        # Token set ratio (handles word order differences)
        token_ratio = fuzz.token_set_ratio(query.lower(), text.lower()) / 100.0
        
        return max(fuzzy_ratio, token_ratio)
    
    def semantic_relevance_score(self, query_keywords: List[str], item_keywords: List[str]) -> float:
        """Calculate semantic relevance between query and item keywords"""
        if not query_keywords or not item_keywords:
            return 0.0
        
        query_set = set(query_keywords)
        item_set = set(item_keywords)
        
        # Jaccard similarity
        intersection = len(query_set & item_set)
        union = len(query_set | item_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_comprehensive_score(self, item, query: str, query_keywords: List[str]) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive relevance score with breakdown"""
        scores = {}
        
        # Quality score (0.0 - 1.0)
        quality_score = self.quality_scorer.score_prompt_quality(item)
        scores['quality'] = quality_score
        
        # Title relevance (high weight)
        title_score = self.fuzzy_score(query, item.title)
        scores['title'] = title_score
        
        # Description relevance
        desc_score = self.fuzzy_score(query, item.description)
        scores['description'] = desc_score
        
        # Content relevance
        content_score = self.fuzzy_score(query, item.content[:1000])  # First 1000 chars
        scores['content'] = content_score
        
        # Tag relevance
        tag_text = ' '.join(item.tags)
        tag_score = self.fuzzy_score(query, tag_text)
        scores['tags'] = tag_score
        
        # Keyword semantic relevance
        item_keywords = self.extract_keywords(f"{item.title} {item.description} {' '.join(item.tags)}")
        semantic_score = self.semantic_relevance_score(query_keywords, item_keywords)
        scores['semantic'] = semantic_score
        
        # Category bonus for specific searches
        category_score = 0.0
        if any(cat.lower() in query.lower() for cat in ['guide', 'tutorial', 'security', 'jailbreak', 'system']):
            if item.category.lower() in query.lower():
                category_score = 0.1
        scores['category'] = category_score
        
        # Weighted final score
        final_score = sum(scores[key] * self.weights[key] for key in self.weights)
        
        return final_score, scores
    
    def advanced_search(self, items: List, query: str = "", 
                       min_quality: float = 0.0, max_results: int = 50,
                       category_filter: str = "", tag_filter: List[str] = None,
                       sort_by: str = "relevance", meta_category_filter: str = "",
                       meta_subcategory_filter: str = "") -> Tuple[List, Dict]:
        """
        Advanced search with quality filtering and detailed scoring
        
        Args:
            items: List of items to search
            query: Search query
            min_quality: Minimum quality score (0.0-1.0)
            max_results: Maximum number of results
            category_filter: Category to filter by
            tag_filter: List of tags to filter by
            sort_by: Sort method ('relevance', 'quality', 'title', 'newest')
        """
        if tag_filter is None:
            tag_filter = []
        
        # Extract query keywords
        query_keywords = self.extract_keywords(query) if query else []
        
        # Filter by category
        filtered_items = items
        if category_filter:
            filtered_items = [item for item in filtered_items 
                            if item.category.lower() == category_filter.lower()]
        
        # Filter by meta-category
        if meta_category_filter:
            filtered_items = [item for item in filtered_items 
                            if hasattr(item, 'meta_category') and item.meta_category == meta_category_filter]
        
        # Filter by meta-subcategory
        if meta_subcategory_filter:
            filtered_items = [item for item in filtered_items 
                            if hasattr(item, 'meta_subcategory') and item.meta_subcategory == meta_subcategory_filter]
        
        # Filter by tags
        if tag_filter:
            filtered_items = [item for item in filtered_items 
                            if any(tag.lower() in [t.lower() for t in item.tags] for tag in tag_filter)]
        
        # Score all items
        scored_items = []
        quality_scores = []
        relevance_scores = []
        
        for item in filtered_items:
            if query:
                final_score, score_breakdown = self.calculate_comprehensive_score(item, query, query_keywords)
                quality_score = score_breakdown['quality']
            else:
                # No query - just use quality score
                quality_score = self.quality_scorer.score_prompt_quality(item)
                final_score = quality_score
                score_breakdown = {'quality': quality_score}
            
            # Filter by minimum quality
            if quality_score >= min_quality:
                scored_items.append((final_score, quality_score, item, score_breakdown))
                quality_scores.append(quality_score)
                relevance_scores.append(final_score)
        
        # Sort results
        if sort_by == "quality":
            scored_items.sort(key=lambda x: x[1], reverse=True)  # Sort by quality score
        elif sort_by == "title":
            scored_items.sort(key=lambda x: x[2].title.lower())
        elif sort_by == "newest":
            scored_items.sort(key=lambda x: x[2].created_date or "", reverse=True)
        else:  # relevance (default)
            scored_items.sort(key=lambda x: x[0], reverse=True)  # Sort by final score
        
        # Prepare results
        results = [item for _, _, item, _ in scored_items[:max_results]]
        
        # Prepare search statistics
        stats = {
            'total_found': len(scored_items),
            'returned': len(results),
            'avg_quality': np.mean(quality_scores) if quality_scores else 0.0,
            'avg_relevance': np.mean(relevance_scores) if relevance_scores else 0.0,
            'quality_distribution': {
                'high': len([s for s in quality_scores if s >= 0.7]),
                'medium': len([s for s in quality_scores if 0.4 <= s < 0.7]),
                'low': len([s for s in quality_scores if s < 0.4])
            },
            'query_keywords': query_keywords,
            'filters_applied': {
                'category': category_filter,
                'tags': tag_filter,
                'min_quality': min_quality
            }
        }
        
        return results, stats
    
    def suggest_related_queries(self, query: str, items: List) -> List[str]:
        """Suggest related search queries based on content"""
        if not query:
            return []
        
        query_keywords = self.extract_keywords(query)
        if not query_keywords:
            return []
        
        # Find items that match current query
        relevant_items = []
        for item in items[:100]:  # Sample first 100 items for performance
            if any(keyword in self.extract_keywords(f"{item.title} {item.description}") 
                   for keyword in query_keywords):
                relevant_items.append(item)
        
        # Extract common keywords from relevant items
        all_keywords = []
        for item in relevant_items:
            item_keywords = self.extract_keywords(f"{item.title} {item.description}")
            all_keywords.extend(item_keywords)
        
        # Find frequent keywords not in original query
        keyword_counts = Counter(all_keywords)
        suggestions = []
        
        for keyword, count in keyword_counts.most_common(10):
            if keyword not in query_keywords and count >= 2:
                suggestions.append(f"{query} {keyword}")
        
        return suggestions[:5]
    
    def analyze_search_quality(self, results: List, query: str) -> Dict:
        """Analyze the quality of search results"""
        if not results:
            return {'quality': 'no_results'}
        
        # Calculate quality metrics
        quality_scores = [self.quality_scorer.score_prompt_quality(item) for item in results]
        
        avg_quality = np.mean(quality_scores)
        
        analysis = {
            'result_count': len(results),
            'average_quality': avg_quality,
            'high_quality_count': len([s for s in quality_scores if s >= 0.7]),
            'quality_rating': 'excellent' if avg_quality >= 0.7 else 
                             'good' if avg_quality >= 0.5 else 
                             'fair' if avg_quality >= 0.3 else 'poor',
            'recommendations': []
        }
        
        # Generate recommendations
        if avg_quality < 0.5:
            analysis['recommendations'].append("Try more specific keywords")
            analysis['recommendations'].append("Filter by high-quality categories")
        
        if len(results) < 5 and query:
            analysis['recommendations'].append("Try broader search terms")
            analysis['recommendations'].append("Remove some filters")
        
        return analysis
    
    def update_config(self, config: Dict[str, Any]):
        """Update stop words and scoring weights at runtime."""
        if 'stop_words' in config and isinstance(config['stop_words'], (list, set)):
            self.stop_words = set(config['stop_words'])
        if 'weights' in config and isinstance(config['weights'], dict):
            for k, v in config['weights'].items():
                if k in self.weights:
                    self.weights[k] = float(v)

    def get_config(self) -> Dict[str, Any]:
        """Return current search configuration."""
        return {'stop_words': list(self.stop_words), 'weights': self.weights}
    
    def advanced_search_details(self, items: List, query: str = "",
                                min_quality: float = 0.0, max_results: int = 50,
                                category_filter: str = "", tag_filter: List[str] = None,
                                sort_by: str = "relevance", meta_category_filter: str = "",
                                meta_subcategory_filter: str = "") -> Tuple[List[Dict], Dict]:
        """
        Same as advanced_search but returns per-item score breakdown.
        """
        if tag_filter is None:
            tag_filter = []

        results, stats = self.advanced_search(
            items=items,
            query=query,
            min_quality=min_quality,
            max_results=max_results,
            category_filter=category_filter,
            tag_filter=tag_filter,
            sort_by=sort_by,
            meta_category_filter=meta_category_filter,
            meta_subcategory_filter=meta_subcategory_filter
        )

        query_keywords = self.extract_keywords(query) if query else []
        details = []
        for item in results:
            final_score, breakdown = self.calculate_comprehensive_score(item, query, query_keywords)
            details.append({
                "item": item,
                "score": final_score,
                "breakdown": breakdown
            })
        return details, stats

def create_enhanced_search_engine(config: Optional[dict] = None):
    """Factory function to create enhanced search engine"""
    return EnhancedSearchEngine(config)