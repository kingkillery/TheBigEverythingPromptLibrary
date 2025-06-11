# Claude RAG Implementation: Comprehensive Guide for Retrieval-Augmented Generation

*Source: Anthropic Cookbook - RAG with Claude*  
*Date: June 10, 2025*

## Overview

This comprehensive guide demonstrates how to build production-ready Retrieval-Augmented Generation (RAG) systems using Claude. Based on proven techniques from Anthropic's official cookbook, this guide covers everything from basic RAG implementation to advanced optimization strategies.

## Table of Contents

1. [RAG Fundamentals](#rag-fundamentals)
2. [Document Processing Pipeline](#document-processing-pipeline)
3. [Embedding and Vector Search](#embedding-and-vector-search)
4. [Claude Integration Patterns](#claude-integration-patterns)
5. [Advanced RAG Techniques](#advanced-rag-techniques)
6. [Evaluation and Optimization](#evaluation-and-optimization)
7. [Production Deployment](#production-deployment)

## RAG Fundamentals

### What is Retrieval-Augmented Generation?

RAG combines the power of large language models with external knowledge retrieval to provide accurate, up-to-date, and contextually relevant responses. It addresses the limitations of pre-trained models by dynamically incorporating relevant information during generation.

### Core RAG Components

```python
import anthropic
import numpy as np
from typing import List, Dict, Any
import json

class RAGSystem:
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Add a document to the RAG system."""
        self.documents.append(text)
        self.metadata.append(metadata or {})
        
        # Generate embedding for the document
        embedding = self.generate_embedding(text)
        self.embeddings.append(embedding)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using a suitable embedding model."""
        # This would typically use a dedicated embedding service
        # For demonstration, we'll use a placeholder
        return np.random.rand(768).tolist()  # Replace with actual embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant documents for a query."""
        query_embedding = self.generate_embedding(query)
        
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        
        return results
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norms if norms != 0 else 0
```

## Document Processing Pipeline

### Intelligent Document Chunking

Effective RAG requires smart document segmentation that preserves semantic coherence:

```python
import re
from typing import List, Tuple

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Chunk document by headers while preserving structure."""
        
        # Define header patterns (H1, H2, H3, etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        chunks = []
        current_chunk = ""
        current_header = ""
        current_level = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Save previous chunk if it exists
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'header': current_header,
                        'level': current_level,
                        'word_count': len(current_chunk.split())
                    })
                
                # Start new chunk
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                current_header = header_text
                current_level = header_level
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
                
                # Check if chunk is getting too large
                if len(current_chunk) > self.chunk_size:
                    # Find a good breaking point
                    break_point = self.find_sentence_break(current_chunk, self.chunk_size)
                    
                    chunks.append({
                        'text': current_chunk[:break_point].strip(),
                        'header': current_header,
                        'level': current_level,
                        'word_count': len(current_chunk[:break_point].split())
                    })
                    
                    # Start next chunk with overlap
                    current_chunk = current_chunk[break_point - self.overlap:]
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'header': current_header,
                'level': current_level,
                'word_count': len(current_chunk.split())
            })
        
        return chunks
    
    def find_sentence_break(self, text: str, max_pos: int) -> int:
        """Find the best sentence break point before max_pos."""
        
        # Look for sentence endings before max_pos
        sentence_endings = ['.', '!', '?']
        
        for i in range(min(max_pos, len(text)) - 1, -1, -1):
            if text[i] in sentence_endings and i < len(text) - 1 and text[i + 1].isspace():
                return i + 1
        
        # If no sentence break found, break at word boundary
        for i in range(min(max_pos, len(text)) - 1, -1, -1):
            if text[i].isspace():
                return i
        
        return max_pos
    
    def semantic_chunking(self, text: str, embedding_function) -> List[Dict[str, Any]]:
        """Advanced semantic chunking based on content similarity."""
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_chunk_embeddings = []
        
        for sentence in sentences:
            sentence_embedding = embedding_function(sentence)
            
            if not current_chunk_sentences:
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(sentence_embedding)
                continue
            
            # Calculate semantic similarity with current chunk
            chunk_embedding = np.mean(current_chunk_embeddings, axis=0)
            similarity = self.cosine_similarity(sentence_embedding, chunk_embedding)
            
            # If similarity is high and chunk isn't too large, add to current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            if similarity > 0.7 and len(chunk_text) < self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(sentence_embedding)
            else:
                # Start new chunk
                chunks.append({
                    'text': chunk_text,
                    'sentences': len(current_chunk_sentences),
                    'semantic_coherence': np.mean([
                        self.cosine_similarity(emb, np.mean(current_chunk_embeddings, axis=0))
                        for emb in current_chunk_embeddings
                    ])
                })
                
                current_chunk_sentences = [sentence]
                current_chunk_embeddings = [sentence_embedding]
        
        # Add final chunk
        if current_chunk_sentences:
            chunks.append({
                'text': ' '.join(current_chunk_sentences),
                'sentences': len(current_chunk_sentences),
                'semantic_coherence': np.mean([
                    self.cosine_similarity(emb, np.mean(current_chunk_embeddings, axis=0))
                    for emb in current_chunk_embeddings
                ])
            })
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # This is a simplified version - consider using spaCy or NLTK for production
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
```

## Embedding and Vector Search

### Advanced Embedding Strategies

```python
import faiss
import pickle
from datetime import datetime

class VectorDatabase:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.documents = []
        self.metadata = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        """Add documents with their embeddings to the vector database."""
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search for similar documents."""
        
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        similarities, indices = self.index.search(query_array, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= threshold:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'rank': i + 1,
                    'index': int(idx)
                })
        
        return results
    
    def save(self, filepath: str):
        """Save the vector database to disk."""
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'embeddings': self.embeddings,
            'dimension': self.dimension,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector database from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.embeddings = data['embeddings']
        self.dimension = data['dimension']
```

### Hybrid Search Implementation

```python
from rank_bm25 import BM25Okapi
import string

class HybridSearch:
    def __init__(self, vector_db: VectorDatabase, alpha: float = 0.7):
        self.vector_db = vector_db
        self.alpha = alpha  # Weight for vector search vs BM25
        self.bm25 = None
        self.tokenized_docs = []
    
    def build_bm25_index(self, documents: List[str]):
        """Build BM25 index for keyword search."""
        
        # Tokenize documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower().split()
    
    def search(self, query: str, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """Perform hybrid search combining vector and BM25 search."""
        
        # Vector search
        vector_results = self.vector_db.search(query_embedding, top_k=top_k * 2)
        
        # BM25 search
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores
        combined_results = {}
        
        # Add vector results
        for result in vector_results:
            idx = result['index']
            combined_results[idx] = {
                'document': result['document'],
                'metadata': result['metadata'],
                'vector_score': result['similarity'],
                'bm25_score': bm25_scores[idx] if idx < len(bm25_scores) else 0,
                'index': idx
            }
        
        # Add top BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[-top_k * 2:][::-1]
        for idx in bm25_top_indices:
            if idx not in combined_results:
                combined_results[idx] = {
                    'document': self.vector_db.documents[idx],
                    'metadata': self.vector_db.metadata[idx],
                    'vector_score': 0,
                    'bm25_score': bm25_scores[idx],
                    'index': idx
                }
        
        # Calculate combined scores
        for result in combined_results.values():
            # Normalize scores (simple min-max normalization)
            normalized_vector = result['vector_score']  # Already normalized
            normalized_bm25 = result['bm25_score'] / max(bm25_scores) if max(bm25_scores) > 0 else 0
            
            result['combined_score'] = (
                self.alpha * normalized_vector + 
                (1 - self.alpha) * normalized_bm25
            )
        
        # Sort by combined score and return top-k
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        return sorted_results[:top_k]
```

## Claude Integration Patterns

### Advanced RAG Prompting

```python
class ClaudeRAGPrompts:
    @staticmethod
    def create_rag_prompt(query: str, contexts: List[str], conversation_history: List[Dict] = None) -> str:
        """Create a comprehensive RAG prompt for Claude."""
        
        # Format contexts
        contexts_text = ""
        for i, context in enumerate(contexts, 1):
            contexts_text += f"<context_{i}>\n{context}\n</context_{i}>\n\n"
        
        # Format conversation history if provided
        history_text = ""
        if conversation_history:
            history_text = "<conversation_history>\n"
            for turn in conversation_history[-5:]:  # Last 5 turns
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                history_text += f"{role}: {content}\n"
            history_text += "</conversation_history>\n\n"
        
        prompt = f"""
{history_text}<task>
You are an expert assistant helping users find information from a knowledge base.
Use the provided contexts to answer the user's question accurately and comprehensively.
</task>

<contexts>
{contexts_text}
</contexts>

<question>
{query}
</question>

<instructions>
1. Carefully read through all provided contexts
2. Identify which contexts are most relevant to the question
3. Synthesize information from multiple contexts when appropriate
4. Provide a clear, accurate, and helpful response
5. If the contexts don't contain enough information to answer fully, say so clearly
6. Cite which contexts you used by referencing context_1, context_2, etc.
7. If there are conflicting information in the contexts, acknowledge this
</instructions>

<response_format>
Provide your response in this format:
- **Answer**: [Your comprehensive answer]
- **Sources**: [Which contexts were used, e.g., "Based on context_1 and context_3"]
- **Confidence**: [High/Medium/Low based on available information]
</response_format>
"""
        
        return prompt
    
    @staticmethod
    def create_summarization_prompt(contexts: List[str], focus_area: str = None) -> str:
        """Create a prompt for summarizing retrieved contexts."""
        
        contexts_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        focus_instruction = ""
        if focus_area:
            focus_instruction = f"Pay special attention to information related to: {focus_area}"
        
        prompt = f"""
<task>
Summarize the key information from the following documents.
{focus_instruction}
</task>

<documents>
{contexts_text}
</documents>

<instructions>
1. Identify the main themes and key points across all documents
2. Organize information logically
3. Highlight any important patterns or relationships
4. Note any contradictions or disagreements between documents
5. Keep the summary comprehensive but concise
</instructions>

Please provide a structured summary of the key information.
"""
        
        return prompt
```

### Multi-Step RAG Reasoning

```python
class MultiStepRAG:
    def __init__(self, claude_client, vector_db: VectorDatabase, embedding_function):
        self.client = claude_client
        self.vector_db = vector_db
        self.embed = embedding_function
    
    def answer_complex_query(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Handle complex queries that require multiple retrieval steps."""
        
        steps = []
        final_contexts = []
        
        current_query = query
        
        for iteration in range(max_iterations):
            # Step 1: Retrieve relevant contexts
            query_embedding = self.embed(current_query)
            retrieved_docs = self.vector_db.search(query_embedding, top_k=5)
            
            if not retrieved_docs:
                break
            
            contexts = [doc['document'] for doc in retrieved_docs]
            
            # Step 2: Analyze if we have enough information
            analysis_prompt = f"""
            <query>{query}</query>
            <current_contexts>
            {chr(10).join(contexts)}
            </current_contexts>
            
            Analyze whether the provided contexts contain sufficient information to answer the query.
            If not, what specific additional information is needed?
            
            Respond with:
            - SUFFICIENT: if contexts provide enough information
            - INSUFFICIENT: [specific information needed] if more information is required
            """
            
            analysis_response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=200,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            analysis = analysis_response.content[0].text.strip()
            
            steps.append({
                'iteration': iteration + 1,
                'query': current_query,
                'retrieved_docs': len(retrieved_docs),
                'analysis': analysis
            })
            
            if analysis.startswith("SUFFICIENT"):
                final_contexts.extend(contexts)
                break
            elif analysis.startswith("INSUFFICIENT"):
                # Extract what additional information is needed
                needed_info = analysis.split("INSUFFICIENT:")[1].strip()
                current_query = f"Find information about: {needed_info}"
                final_contexts.extend(contexts)
            else:
                final_contexts.extend(contexts)
                break
        
        # Generate final answer
        final_prompt = ClaudeRAGPrompts.create_rag_prompt(query, final_contexts)
        
        final_response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": final_prompt}]
        )
        
        return {
            'answer': final_response.content[0].text,
            'steps': steps,
            'total_contexts': len(final_contexts),
            'iterations': len(steps)
        }
```

## Advanced RAG Techniques

### Query Enhancement and Rewriting

```python
class QueryEnhancer:
    def __init__(self, claude_client):
        self.client = claude_client
    
    def enhance_query(self, original_query: str, conversation_context: List[str] = None) -> List[str]:
        """Generate multiple enhanced versions of a query for better retrieval."""
        
        context_text = ""
        if conversation_context:
            context_text = f"<conversation_context>\n{chr(10).join(conversation_context)}\n</conversation_context>\n"
        
        prompt = f"""
{context_text}<task>
Generate 3-5 enhanced versions of the user's query to improve information retrieval.
Consider different phrasings, synonyms, and related concepts.
</task>

<original_query>
{original_query}
</original_query>

<instructions>
1. Keep the core intent of the original query
2. Use different phrasings and synonyms
3. Consider both specific and general versions
4. Include technical and layman terms where appropriate
5. Each enhanced query should be on a separate line
</instructions>

Enhanced queries:
"""
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        enhanced_queries = [
            line.strip() 
            for line in response.content[0].text.split('\n') 
            if line.strip() and not line.strip().startswith('Enhanced queries:')
        ]
        
        return [original_query] + enhanced_queries
    
    def decompose_complex_query(self, complex_query: str) -> List[str]:
        """Break down complex queries into simpler sub-queries."""
        
        prompt = f"""
<task>
Break down this complex query into 2-4 simpler sub-queries that together would help answer the original question.
</task>

<complex_query>
{complex_query}
</complex_query>

<instructions>
1. Identify the key components of the complex query
2. Create focused sub-queries for each component
3. Ensure sub-queries are specific and searchable
4. Each sub-query should be on a separate line
5. Number each sub-query (1., 2., etc.)
</instructions>

Sub-queries:
"""
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        sub_queries = []
        for line in response.content[0].text.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Remove numbering
                sub_query = line.split('.', 1)[1].strip()
                sub_queries.append(sub_query)
        
        return sub_queries
```

### Context Re-ranking

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ContextReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def rerank_contexts(self, query: str, contexts: List[Dict], top_k: int = 5) -> List[Dict]:
        """Re-rank retrieved contexts using a cross-encoder model."""
        
        scores = []
        
        for context in contexts:
            # Prepare input for cross-encoder
            text = context['document']
            inputs = self.tokenizer(
                query, 
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Get relevance score
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item()
            
            scores.append(score)
        
        # Combine original similarity with re-ranking score
        for i, context in enumerate(contexts):
            context['rerank_score'] = scores[i]
            context['combined_score'] = (context.get('similarity', 0.5) + scores[i]) / 2
        
        # Sort by combined score
        reranked = sorted(contexts, key=lambda x: x['combined_score'], reverse=True)
        
        return reranked[:top_k]
```

## Evaluation and Optimization

### Comprehensive RAG Evaluation

```python
class RAGEvaluator:
    def __init__(self, claude_client):
        self.client = claude_client
    
    def evaluate_retrieval(self, test_queries: List[str], ground_truth_docs: List[List[str]], 
                          rag_system, top_k: int = 5) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        
        for query, relevant_docs in zip(test_queries, ground_truth_docs):
            # Get retrieval results
            query_embedding = rag_system.embed(query)
            retrieved = rag_system.vector_db.search(query_embedding, top_k=top_k)
            retrieved_docs = [doc['document'] for doc in retrieved]
            
            # Calculate metrics
            precision = self.calculate_precision(retrieved_docs, relevant_docs)
            recall = self.calculate_recall(retrieved_docs, relevant_docs)
            mrr = self.calculate_mrr(retrieved_docs, relevant_docs)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
        
        return {
            'precision@k': np.mean(precision_scores),
            'recall@k': np.mean(recall_scores),
            'mrr': np.mean(mrr_scores),
            'f1@k': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                   (np.mean(precision_scores) + np.mean(recall_scores))
        }
    
    def calculate_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate precision@k."""
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(retrieved)
    
    def calculate_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate recall@k."""
        if not relevant:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(relevant)
    
    def calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_generation_quality(self, queries: List[str], contexts: List[List[str]], 
                                   generated_answers: List[str]) -> Dict[str, float]:
        """Evaluate generation quality using Claude as a judge."""
        
        relevance_scores = []
        accuracy_scores = []
        completeness_scores = []
        
        for query, context_list, answer in zip(queries, contexts, generated_answers):
            evaluation_prompt = f"""
<task>
Evaluate the quality of this RAG system response across three dimensions.
</task>

<query>
{query}
</query>

<contexts>
{chr(10).join(context_list)}
</contexts>

<generated_answer>
{answer}
</generated_answer>

<evaluation_criteria>
Rate each dimension from 1-10:

1. RELEVANCE: How well does the answer address the specific query?
2. ACCURACY: How factually correct is the answer based on the contexts?
3. COMPLETENESS: How comprehensive is the answer given the available information?
</evaluation_criteria>

<format>
Provide scores in this exact format:
RELEVANCE: [score]
ACCURACY: [score]  
COMPLETENESS: [score]
EXPLANATION: [brief explanation of scores]
</format>
"""
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            
            # Parse scores
            scores = self.parse_evaluation_scores(response.content[0].text)
            relevance_scores.append(scores.get('relevance', 5))
            accuracy_scores.append(scores.get('accuracy', 5))
            completeness_scores.append(scores.get('completeness', 5))
        
        return {
            'relevance': np.mean(relevance_scores),
            'accuracy': np.mean(accuracy_scores),
            'completeness': np.mean(completeness_scores),
            'overall': np.mean([
                np.mean(relevance_scores),
                np.mean(accuracy_scores),
                np.mean(completeness_scores)
            ])
        }
    
    def parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse evaluation scores from Claude's response."""
        scores = {}
        
        for line in evaluation_text.split('\n'):
            line = line.strip()
            if ':' in line:
                metric, score_str = line.split(':', 1)
                metric = metric.strip().lower()
                
                try:
                    score = float(score_str.strip())
                    scores[metric] = score
                except ValueError:
                    continue
        
        return scores
```

## Production Deployment

### Caching and Performance Optimization

```python
import redis
import hashlib
from functools import wraps
import time

class RAGCache:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 default_ttl: int = 3600):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.default_ttl = default_ttl
    
    def cache_key(self, query: str, context_hash: str = None) -> str:
        """Generate cache key for query and context."""
        key_data = f"query:{query}"
        if context_hash:
            key_data += f":context:{context_hash}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_response(self, query: str, context_hash: str = None) -> Dict:
        """Get cached response for query."""
        cache_key = self.cache_key(query, context_hash)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        return None
    
    def cache_response(self, query: str, response: Dict, context_hash: str = None, 
                      ttl: int = None):
        """Cache response for query."""
        cache_key = self.cache_key(query, context_hash)
        response_data = {
            'response': response,
            'timestamp': time.time()
        }
        
        ttl = ttl or self.default_ttl
        self.redis_client.setex(
            cache_key, 
            ttl, 
            json.dumps(response_data)
        )

class ProductionRAGSystem:
    def __init__(self, claude_client, vector_db: VectorDatabase, 
                 embedding_function, cache: RAGCache = None):
        self.client = claude_client
        self.vector_db = vector_db
        self.embed = embedding_function
        self.cache = cache
        self.query_enhancer = QueryEnhancer(claude_client)
    
    def answer_query(self, query: str, use_cache: bool = True, 
                    enhance_query: bool = True) -> Dict[str, Any]:
        """Production-ready query answering with caching and enhancement."""
        
        start_time = time.time()
        
        # Check cache first
        if use_cache and self.cache:
            cached_response = self.cache.get_cached_response(query)
            if cached_response:
                cached_response['cache_hit'] = True
                cached_response['response_time'] = time.time() - start_time
                return cached_response
        
        # Enhance query if requested
        queries_to_search = [query]
        if enhance_query:
            enhanced_queries = self.query_enhancer.enhance_query(query)
            queries_to_search.extend(enhanced_queries)
        
        # Retrieve contexts for all query variations
        all_contexts = []
        for q in queries_to_search:
            q_embedding = self.embed(q)
            contexts = self.vector_db.search(q_embedding, top_k=3)
            all_contexts.extend(contexts)
        
        # Remove duplicates and re-rank
        unique_contexts = []
        seen_docs = set()
        for ctx in all_contexts:
            doc_hash = hashlib.md5(ctx['document'].encode()).hexdigest()
            if doc_hash not in seen_docs:
                unique_contexts.append(ctx)
                seen_docs.add(doc_hash)
        
        # Take top contexts
        unique_contexts = sorted(unique_contexts, key=lambda x: x['similarity'], reverse=True)[:5]
        
        # Generate answer
        context_docs = [ctx['document'] for ctx in unique_contexts]
        prompt = ClaudeRAGPrompts.create_rag_prompt(query, context_docs)
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = {
            'answer': response.content[0].text,
            'contexts_used': len(unique_contexts),
            'query_variations': len(queries_to_search),
            'cache_hit': False,
            'response_time': time.time() - start_time,
            'contexts': unique_contexts
        }
        
        # Cache the result
        if use_cache and self.cache:
            context_hash = hashlib.md5(''.join(context_docs).encode()).hexdigest()
            self.cache.cache_response(query, result, context_hash)
        
        return result
```

### Monitoring and Analytics

```python
import logging
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class RAGMetrics:
    query: str
    response_time: float
    contexts_retrieved: int
    cache_hit: bool
    user_feedback: Optional[float] = None
    timestamp: float = None

class RAGMonitor:
    def __init__(self, log_file: str = "rag_metrics.log"):
        self.logger = logging.getLogger("RAGMonitor")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.metrics = []
    
    def log_query(self, metrics: RAGMetrics):
        """Log query metrics."""
        if metrics.timestamp is None:
            metrics.timestamp = time.time()
        
        self.metrics.append(metrics)
        
        log_data = {
            'query': metrics.query,
            'response_time': metrics.response_time,
            'contexts_retrieved': metrics.contexts_retrieved,
            'cache_hit': metrics.cache_hit,
            'user_feedback': metrics.user_feedback,
            'timestamp': metrics.timestamp
        }
        
        self.logger.info(json.dumps(log_data))
    
    def get_performance_stats(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get performance statistics for the last time window (in seconds)."""
        
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics 
            if current_time - m.timestamp <= time_window
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'total_queries': len(recent_metrics),
            'avg_response_time': np.mean([m.response_time for m in recent_metrics]),
            'cache_hit_rate': np.mean([m.cache_hit for m in recent_metrics]),
            'avg_contexts_retrieved': np.mean([m.contexts_retrieved for m in recent_metrics]),
            'avg_user_feedback': np.mean([
                m.user_feedback for m in recent_metrics 
                if m.user_feedback is not None
            ]) if any(m.user_feedback is not None for m in recent_metrics) else None
        }
```

## Best Practices Summary

### Document Processing
1. **Semantic Chunking**: Preserve meaning while maintaining searchable chunk sizes
2. **Metadata Enrichment**: Include headers, document types, and structural information
3. **Quality Control**: Filter and validate documents before indexing

### Retrieval Optimization
1. **Hybrid Search**: Combine vector and keyword search for better coverage
2. **Query Enhancement**: Use multiple query variations and decomposition
3. **Re-ranking**: Apply cross-encoder models for better relevance

### Generation Quality
1. **Structured Prompts**: Use clear XML tags and instructions
2. **Citation Requirements**: Always cite sources and provide confidence levels
3. **Context Management**: Balance context length with relevance

### Production Considerations
1. **Caching Strategy**: Implement intelligent caching for common queries
2. **Monitoring**: Track performance metrics and user satisfaction
3. **Error Handling**: Graceful degradation when retrieval or generation fails
4. **Scalability**: Design for concurrent users and large document collections

This comprehensive guide provides a complete framework for implementing production-ready RAG systems using Claude, from basic retrieval to advanced optimization techniques with full monitoring and evaluation capabilities.