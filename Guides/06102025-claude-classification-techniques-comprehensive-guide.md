# Claude Classification Techniques: Comprehensive Implementation Guide

*Source: Anthropic Cookbook - Classification with Claude*  
*Date: June 10, 2025*

## Overview

This comprehensive guide demonstrates how to implement robust classification systems using Claude, based on proven techniques from Anthropic's official cookbook. The methodology covers everything from basic classification to advanced RAG-enhanced approaches with proper evaluation frameworks.

## Table of Contents

1. [Classification Fundamentals](#classification-fundamentals)
2. [Prompt Engineering for Classification](#prompt-engineering-for-classification)
3. [RAG-Enhanced Classification](#rag-enhanced-classification)
4. [Evaluation and Testing](#evaluation-and-testing)
5. [Production Implementation](#production-implementation)
6. [Advanced Techniques](#advanced-techniques)

## Classification Fundamentals

### Core Classification Components

Claude classification systems require three essential components:

1. **Clear Category Definitions**: Precise, unambiguous category descriptions
2. **Contextual Examples**: Representative examples for each category
3. **Evaluation Framework**: Systematic testing and validation

### Basic Classification Pattern

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

def basic_classification(text, categories):
    """Basic classification with clear category definitions."""
    
    categories_text = "\n".join([f"- {cat}" for cat in categories])
    
    prompt = f"""
    Classify the following text into one of these categories:
    
    {categories_text}
    
    Text to classify: {text}
    
    Respond with only the category name.
    """
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()
```

## Prompt Engineering for Classification

### Enhanced Category Descriptions

Effective classification requires detailed category descriptions with context:

```python
def enhanced_classification_prompt(text, category_definitions):
    """Classification with detailed category descriptions."""
    
    categories_section = ""
    for category, description in category_definitions.items():
        categories_section += f"**{category}**: {description}\n"
    
    prompt = f"""
    <task>
    Classify the following support ticket into the most appropriate category.
    </task>
    
    <categories>
    {categories_section}
    </categories>
    
    <examples>
    Example 1:
    Text: "My car insurance premium went up without explanation"
    Category: Billing Inquiry
    
    Example 2:
    Text: "I was in an accident and need to file a claim"
    Category: Claims Processing
    
    Example 3:
    Text: "I want to add my teenage daughter to my policy"
    Category: Policy Modification
    </examples>
    
    <ticket>
    {text}
    </ticket>
    
    <instructions>
    1. Read the ticket carefully
    2. Consider which category best matches the customer's primary concern
    3. Respond with only the category name
    4. If uncertain, choose the closest match
    </instructions>
    
    Category:
    """
    
    return prompt
```

### Multi-Shot Learning Approach

```python
def create_multishot_classification_prompt(text, examples_by_category):
    """Create a multi-shot learning prompt with examples for each category."""
    
    examples_section = ""
    for category, examples in examples_by_category.items():
        examples_section += f"\n**{category} Examples:**\n"
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples per category
            examples_section += f"{i}. \"{example}\"\n"
    
    prompt = f"""
    <task>
    You are an expert at classifying insurance support tickets. Use the examples below to understand each category, then classify the new ticket.
    </task>
    
    <examples>
    {examples_section}
    </examples>
    
    <new_ticket>
    {text}
    </new_ticket>
    
    <instructions>
    Based on the examples above, classify this new ticket into the most appropriate category.
    Consider the primary intent and concern of the customer.
    Respond with only the category name.
    </instructions>
    
    Classification:
    """
    
    return prompt
```

## RAG-Enhanced Classification

### Document Retrieval for Context

RAG enhancement provides relevant context for better classification accuracy:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGClassifier:
    def __init__(self, anthropic_client, embedding_function):
        self.client = anthropic_client
        self.embed = embedding_function
        self.knowledge_base = []
        self.knowledge_embeddings = []
    
    def add_knowledge(self, documents):
        """Add knowledge base documents for RAG."""
        self.knowledge_base.extend(documents)
        new_embeddings = [self.embed(doc) for doc in documents]
        self.knowledge_embeddings.extend(new_embeddings)
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve most relevant context for classification."""
        query_embedding = self.embed(query)
        
        similarities = [
            cosine_similarity([query_embedding], [doc_emb])[0][0]
            for doc_emb in self.knowledge_embeddings
        ]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.knowledge_base[i] for i in top_indices]
    
    def classify_with_rag(self, text, categories):
        """Classify using RAG-enhanced context."""
        
        # Retrieve relevant context
        context_docs = self.retrieve_context(text)
        context = "\n\n".join(context_docs)
        
        prompt = f"""
        <context>
        Relevant information for classification:
        {context}
        </context>
        
        <categories>
        {', '.join(categories)}
        </categories>
        
        <text_to_classify>
        {text}
        </text_to_classify>
        
        <instructions>
        Using the provided context to inform your decision, classify the text into one of the given categories.
        The context provides background information that may help disambiguate the classification.
        
        Think through your reasoning:
        1. What is the main intent of the text?
        2. How does the context inform this classification?
        3. Which category best fits?
        
        Respond with your reasoning followed by the category name.
        </instructions>
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
```

## Evaluation and Testing

### Classification Metrics

Implement comprehensive evaluation using standard classification metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationEvaluator:
    def __init__(self):
        self.predictions = []
        self.true_labels = []
    
    def add_prediction(self, predicted, actual):
        """Add a single prediction for evaluation."""
        self.predictions.append(predicted)
        self.true_labels.append(actual)
    
    def evaluate(self, categories):
        """Generate comprehensive evaluation metrics."""
        
        # Classification report
        report = classification_report(
            self.true_labels, 
            self.predictions, 
            labels=categories,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            self.true_labels, 
            self.predictions, 
            labels=categories
        )
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def plot_confusion_matrix(self, categories, title="Classification Confusion Matrix"):
        """Visualize confusion matrix."""
        cm = confusion_matrix(self.true_labels, self.predictions, labels=categories)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=categories,
            yticklabels=categories
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def per_category_analysis(self, categories):
        """Detailed per-category performance analysis."""
        report = classification_report(
            self.true_labels, 
            self.predictions, 
            labels=categories,
            output_dict=True
        )
        
        analysis = []
        for category in categories:
            if category in report:
                analysis.append({
                    'Category': category,
                    'Precision': report[category]['precision'],
                    'Recall': report[category]['recall'],
                    'F1-Score': report[category]['f1-score'],
                    'Support': report[category]['support']
                })
        
        return pd.DataFrame(analysis)
```

### A/B Testing Framework

```python
import random
from datetime import datetime

class ClassificationABTest:
    def __init__(self, method_a, method_b, test_ratio=0.5):
        self.method_a = method_a
        self.method_b = method_b
        self.test_ratio = test_ratio
        self.results_a = []
        self.results_b = []
    
    def classify_sample(self, text, categories, true_label):
        """Classify a sample using A/B testing."""
        
        # Randomly assign to method A or B
        use_method_a = random.random() < self.test_ratio
        
        if use_method_a:
            prediction = self.method_a(text, categories)
            self.results_a.append({
                'prediction': prediction,
                'true_label': true_label,
                'correct': prediction == true_label,
                'timestamp': datetime.now()
            })
            return prediction, 'A'
        else:
            prediction = self.method_b(text, categories)
            self.results_b.append({
                'prediction': prediction,
                'true_label': true_label,
                'correct': prediction == true_label,
                'timestamp': datetime.now()
            })
            return prediction, 'B'
    
    def get_results(self):
        """Get A/B test results comparison."""
        
        accuracy_a = sum(r['correct'] for r in self.results_a) / len(self.results_a) if self.results_a else 0
        accuracy_b = sum(r['correct'] for r in self.results_b) / len(self.results_b) if self.results_b else 0
        
        return {
            'method_a': {
                'samples': len(self.results_a),
                'accuracy': accuracy_a,
                'results': self.results_a
            },
            'method_b': {
                'samples': len(self.results_b),
                'accuracy': accuracy_b,
                'results': self.results_b
            },
            'improvement': accuracy_b - accuracy_a
        }
```

## Production Implementation

### Concurrent Processing

Handle high-volume classification with concurrent API calls:

```python
import asyncio
import anthropic
from concurrent.futures import ThreadPoolExecutor
import time

class ProductionClassifier:
    def __init__(self, api_key, max_workers=5):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_workers = max_workers
    
    def classify_single(self, text, prompt_template):
        """Classify a single text sample."""
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt_template.format(text=text)}]
            )
            return {
                'text': text,
                'classification': response.content[0].text.strip(),
                'success': True,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'text': text,
                'classification': None,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def classify_batch(self, texts, prompt_template):
        """Classify multiple texts concurrently."""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.classify_single, text, prompt_template)
                for text in texts
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30-second timeout
                    results.append(result)
                except Exception as e:
                    results.append({
                        'text': 'Unknown',
                        'classification': None,
                        'success': False,
                        'error': f"Future error: {str(e)}",
                        'timestamp': time.time()
                    })
        
        return results
    
    def get_batch_stats(self, results):
        """Get statistics for batch processing results."""
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        
        return {
            'total_processed': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'avg_processing_time': sum(r.get('processing_time', 0) for r in results) / total if total > 0 else 0
        }
```

### Error Handling and Retry Logic

```python
import time
import random

class RobustClassifier:
    def __init__(self, client, max_retries=3, base_delay=1):
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def classify_with_retry(self, text, prompt, exponential_backoff=True):
        """Classify with robust error handling and retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return {
                    'classification': response.content[0].text.strip(),
                    'success': True,
                    'attempt': attempt + 1
                }
                
            except anthropic.RateLimitError as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt) if exponential_backoff else self.base_delay
                    delay += random.uniform(0, 1)  # Add jitter
                    time.sleep(delay)
                    continue
                return {'classification': None, 'success': False, 'error': 'Rate limit exceeded'}
                
            except anthropic.APIError as e:
                if attempt < self.max_retries:
                    time.sleep(self.base_delay)
                    continue
                return {'classification': None, 'success': False, 'error': f'API error: {str(e)}'}
                
            except Exception as e:
                return {'classification': None, 'success': False, 'error': f'Unexpected error: {str(e)}'}
        
        return {'classification': None, 'success': False, 'error': 'Max retries exceeded'}
```

## Advanced Techniques

### Hierarchical Classification

```python
class HierarchicalClassifier:
    def __init__(self, client):
        self.client = client
        self.hierarchy = {}
    
    def add_hierarchy_level(self, parent, children):
        """Define hierarchical relationships between categories."""
        self.hierarchy[parent] = children
    
    def classify_hierarchical(self, text):
        """Perform hierarchical classification."""
        
        # Start with top-level classification
        top_level_categories = list(self.hierarchy.keys())
        
        prompt = f"""
        Classify this text into one of these high-level categories:
        {', '.join(top_level_categories)}
        
        Text: {text}
        
        Category:
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        top_level = response.content[0].text.strip()
        
        # If there are subcategories, classify further
        if top_level in self.hierarchy:
            subcategories = self.hierarchy[top_level]
            
            subprompt = f"""
            You've determined this text belongs to: {top_level}
            
            Now classify it more specifically into one of these subcategories:
            {', '.join(subcategories)}
            
            Text: {text}
            
            Subcategory:
            """
            
            subresponse = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                messages=[{"role": "user", "content": subprompt}]
            )
            
            subcategory = subresponse.content[0].text.strip()
            return f"{top_level} > {subcategory}"
        
        return top_level
```

### Confidence Scoring

```python
def classify_with_confidence(client, text, categories):
    """Classify with confidence scoring."""
    
    prompt = f"""
    Classify the following text and provide a confidence score.
    
    Categories: {', '.join(categories)}
    Text: {text}
    
    Provide your response in this exact format:
    Classification: [category]
    Confidence: [score from 0-100]
    Reasoning: [brief explanation]
    """
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result_text = response.content[0].text
    
    # Parse the structured response
    lines = result_text.strip().split('\n')
    
    classification = None
    confidence = None
    reasoning = None
    
    for line in lines:
        if line.startswith('Classification:'):
            classification = line.split('Classification:')[1].strip()
        elif line.startswith('Confidence:'):
            confidence_str = line.split('Confidence:')[1].strip()
            try:
                confidence = int(confidence_str)
            except ValueError:
                confidence = None
        elif line.startswith('Reasoning:'):
            reasoning = line.split('Reasoning:')[1].strip()
    
    return {
        'classification': classification,
        'confidence': confidence,
        'reasoning': reasoning,
        'raw_response': result_text
    }
```

## Best Practices Summary

### Prompt Engineering
1. **Clear Categories**: Provide detailed, unambiguous category descriptions
2. **Examples**: Include representative examples for each category
3. **Context**: Use relevant domain knowledge and background information
4. **Structure**: Organize prompts with clear sections and instructions

### Evaluation
1. **Comprehensive Metrics**: Use precision, recall, F1-score, and confusion matrices
2. **A/B Testing**: Compare different approaches systematically
3. **Error Analysis**: Examine misclassifications to improve the system
4. **Continuous Monitoring**: Track performance over time

### Production Deployment
1. **Concurrent Processing**: Handle high volumes efficiently
2. **Error Handling**: Implement robust retry logic and graceful degradation
3. **Monitoring**: Log performance metrics and error rates
4. **Caching**: Cache common classifications to reduce API calls

### Advanced Features
1. **Hierarchical Classification**: Break complex categories into hierarchies
2. **Confidence Scoring**: Provide uncertainty estimates
3. **RAG Enhancement**: Use retrieval for better context
4. **Adaptive Learning**: Update classifications based on feedback

This comprehensive guide provides a complete framework for implementing production-ready classification systems using Claude, from basic categorization to advanced hierarchical approaches with full evaluation and monitoring capabilities.