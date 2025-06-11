# Hugging Face Open-Source AI: Comprehensive Implementation Guide

*Source: Hugging Face Open-Source AI Cookbook*  
*Date: June 10, 2025*

## Overview

This comprehensive guide demonstrates how to leverage the Hugging Face ecosystem for building advanced AI applications using open-source models. Based on proven techniques from Hugging Face's official cookbook, this guide covers fine-tuning, RAG systems, and deployment strategies that provide cost-effective alternatives to proprietary AI services.

## Table of Contents

1. [Hugging Face Ecosystem Overview](#hugging-face-ecosystem-overview)
2. [Fine-Tuning on Consumer Hardware](#fine-tuning-on-consumer-hardware)
3. [Open-Source RAG Systems](#open-source-rag-systems)
4. [Advanced RAG Techniques](#advanced-rag-techniques)
5. [Agentic AI Systems](#agentic-ai-systems)
6. [Production Deployment](#production-deployment)
7. [Performance Optimization](#performance-optimization)

## Hugging Face Ecosystem Overview

### Core Components and Advantages

The Hugging Face ecosystem provides a comprehensive suite of tools for building AI applications with open-source models:

```python
# Essential imports for Hugging Face ecosystem
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, pipeline
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

class HuggingFaceSetup:
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
    
    def list_popular_models(self, task: str = "text-generation") -> dict:
        """Get popular models for specific tasks."""
        
        model_recommendations = {
            "text-generation": [
                "microsoft/DialoGPT-medium",
                "microsoft/CodeBERT-base",
                "bigcode/starcoderbase-1b",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "codellama/CodeLlama-7b-Python-hf"
            ],
            "embeddings": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
                "thenlper/gte-small",
                "intfloat/e5-base-v2"
            ],
            "classification": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "microsoft/DialoGPT-medium",
                "facebook/bart-large-mnli"
            ],
            "reranking": [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "colbert-ir/colbertv2.0"
            ]
        }
        
        return model_recommendations.get(task, [])
    
    def estimate_model_requirements(self, model_name: str) -> dict:
        """Estimate memory and compute requirements for a model."""
        
        # Simple estimation based on model name patterns
        size_indicators = {
            "7b": {"memory_gb": 14, "vram_gb": 8, "compute": "medium"},
            "1b": {"memory_gb": 4, "vram_gb": 2, "compute": "low"},
            "base": {"memory_gb": 2, "vram_gb": 1, "compute": "low"},
            "large": {"memory_gb": 6, "vram_gb": 4, "compute": "medium"},
            "xl": {"memory_gb": 12, "vram_gb": 8, "compute": "high"}
        }
        
        for size, requirements in size_indicators.items():
            if size in model_name.lower():
                return requirements
        
        return {"memory_gb": 4, "vram_gb": 2, "compute": "unknown"}
```

### Model Selection and Evaluation

```python
class ModelEvaluator:
    def __init__(self):
        self.evaluation_metrics = {}
    
    def compare_models(self, model_names: list, test_prompts: list) -> dict:
        """Compare multiple models on test prompts."""
        
        results = {}
        
        for model_name in model_names:
            try:
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                model_results = []
                
                for prompt in test_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()
                    
                    model_results.append({
                        'prompt': prompt,
                        'response': response,
                        'response_length': len(response.split())
                    })
                
                results[model_name] = {
                    'results': model_results,
                    'avg_response_length': np.mean([r['response_length'] for r in model_results]),
                    'model_size': self.get_model_size(model_name)
                }
                
                # Clean up memory
                del model, tokenizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def get_model_size(self, model_name: str) -> str:
        """Estimate model size category."""
        if any(size in model_name.lower() for size in ['7b', '8b']):
            return 'large'
        elif any(size in model_name.lower() for size in ['1b', '2b']):
            return 'medium'
        else:
            return 'small'
```

## Fine-Tuning on Consumer Hardware

### Parameter-Efficient Fine-Tuning with LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

class ParameterEfficientFineTuner:
    def __init__(self, base_model_name: str, use_quantization: bool = True):
        self.base_model_name = base_model_name
        self.use_quantization = use_quantization
        self.setup_quantization_config()
    
    def setup_quantization_config(self):
        """Configure 4-bit quantization for memory efficiency."""
        
        if self.use_quantization:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16
            )
        else:
            self.bnb_config = None
    
    def setup_lora_config(self, task_type: str = "CAUSAL_LM", r: int = 8, 
                         alpha: int = 32, dropout: float = 0.1):
        """Configure LoRA for parameter-efficient training."""
        
        # Target modules vary by model architecture
        target_modules_map = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "starcoder": ["c_attn", "c_proj", "c_fc"],
            "codellama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
        
        # Detect model architecture
        model_type = self.detect_model_architecture()
        target_modules = target_modules_map.get(model_type, ["q_proj", "v_proj"])
        
        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=getattr(TaskType, task_type),
            target_modules=target_modules
        )
    
    def detect_model_architecture(self) -> str:
        """Detect model architecture from model name."""
        
        model_name_lower = self.base_model_name.lower()
        
        if "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "starcoder" in model_name_lower or "coder" in model_name_lower:
            return "starcoder"
        else:
            return "generic"
    
    def load_model_for_training(self):
        """Load and prepare model for fine-tuning."""
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config if self.use_quantization else None,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.use_quantization else torch.float16,
            trust_remote_code=True
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_param:,} || "
              f"Trainable%: {100 * trainable_params / all_param:.2f}")
    
    def prepare_dataset(self, dataset_name: str = None, custom_data: list = None):
        """Prepare dataset for fine-tuning."""
        
        if custom_data:
            # Use custom data
            dataset = Dataset.from_list(custom_data)
        elif dataset_name:
            # Load from Hugging Face Hub
            dataset = load_dataset(dataset_name, split="train")
        else:
            raise ValueError("Either dataset_name or custom_data must be provided")
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Adjust based on your data format
            if 'text' in examples:
                texts = examples['text']
            elif 'code' in examples:
                texts = examples['code']
            else:
                # Assume first key contains text data
                texts = examples[list(examples.keys())[0]]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        self.tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def train(self, output_dir: str = "./fine-tuned-model", 
              num_train_epochs: int = 3, learning_rate: float = 2e-4):
        """Execute fine-tuning process."""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            gradient_checkpointing=True,
            dataloader_pin_memory=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
    
    def data_collator(self, features):
        """Custom data collator for causal language modeling."""
        
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        
        return batch
```

### Code-Specific Fine-Tuning

```python
class CodeModelFineTuner(ParameterEfficientFineTuner):
    def __init__(self, base_model_name: str = "bigcode/starcoderbase-1b"):
        super().__init__(base_model_name)
        self.setup_lora_config(task_type="CAUSAL_LM", r=8, alpha=32)
    
    def prepare_code_dataset(self, repositories: list = None, languages: list = None):
        """Prepare code dataset for fine-tuning."""
        
        if repositories:
            # Use specific repositories
            code_data = self.extract_code_from_repos(repositories, languages)
        else:
            # Use default code dataset
            dataset = load_dataset("smangrul/hf-stack-v1", split="train")
            code_data = dataset['content']
        
        # Apply Fill-in-the-Middle (FIM) transformation
        fim_data = self.apply_fim_transformation(code_data)
        
        # Create dataset
        dataset = Dataset.from_dict({'text': fim_data})
        
        self.prepare_dataset(custom_data=dataset)
    
    def apply_fim_transformation(self, code_samples: list, fim_rate: float = 0.5):
        """Apply Fill-in-the-Middle transformation for better infilling."""
        
        transformed_samples = []
        
        for code in code_samples:
            if np.random.random() < fim_rate:
                # Apply FIM transformation
                lines = code.split('\n')
                
                if len(lines) > 3:
                    # Choose random middle section to mask
                    start_idx = np.random.randint(1, len(lines) // 2)
                    end_idx = np.random.randint(len(lines) // 2, len(lines) - 1)
                    
                    prefix = '\n'.join(lines[:start_idx])
                    middle = '\n'.join(lines[start_idx:end_idx])
                    suffix = '\n'.join(lines[end_idx:])
                    
                    # FIM format: <fim_prefix>prefix<fim_suffix>suffix<fim_middle>middle
                    fim_sample = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"
                    transformed_samples.append(fim_sample)
                else:
                    transformed_samples.append(code)
            else:
                transformed_samples.append(code)
        
        return transformed_samples
    
    def extract_code_from_repos(self, repositories: list, languages: list = None) -> list:
        """Extract code from specified repositories."""
        
        code_samples = []
        
        for repo in repositories:
            try:
                # This would integrate with git or GitHub API
                # For demonstration, returning placeholder
                code_samples.append(f"# Code from {repo}\n# Placeholder implementation")
            except Exception as e:
                print(f"Error processing {repo}: {e}")
        
        return code_samples
```

## Open-Source RAG Systems

### Vector Database Integration

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any

class OpenSourceRAGSystem:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 llm_model_name: str = "microsoft/DialoGPT-medium"):
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector database (FAISS)
        self.vector_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.document_store = []
        self.metadata_store = []
        
        # Initialize LLM
        self.setup_llm(llm_model_name)
    
    def setup_llm(self, model_name: str):
        """Setup language model for generation."""
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add pad token if missing
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector database."""
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        
        # Add to FAISS index
        self.vector_index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.document_store.extend(documents)
        self.metadata_store.extend(metadata)
        
        print(f"Added {len(documents)} documents. Total: {len(self.document_store)}")
    
    def search_documents(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search for relevant documents."""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index
        similarities, indices = self.vector_index.search(
            query_embedding.astype(np.float32), 
            top_k
        )
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= threshold and idx < len(self.document_store):
                results.append({
                    'document': self.document_store[idx],
                    'metadata': self.metadata_store[idx],
                    'similarity': float(similarity),
                    'index': int(idx)
                })
        
        return results
    
    def generate_response(self, query: str, context_documents: List[str], 
                         max_length: int = 500) -> str:
        """Generate response using retrieved context."""
        
        # Prepare context
        context = "\n\n".join(context_documents[:3])  # Use top 3 documents
        
        # Create prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer based on the context:"""
        
        # Tokenize
        inputs = self.llm_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
    
    def rag_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute complete RAG pipeline."""
        
        # Search for relevant documents
        retrieved_docs = self.search_documents(query, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'query': query,
                'answer': "I don't have enough information to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Extract document texts
        context_docs = [doc['document'] for doc in retrieved_docs]
        
        # Generate response
        answer = self.generate_response(query, context_docs)
        
        return {
            'query': query,
            'answer': answer,
            'sources': retrieved_docs,
            'confidence': np.mean([doc['similarity'] for doc in retrieved_docs])
        }
```

### Alternative Vector Databases

```python
# Integration with Milvus for production scale
from pymilvus import MilvusClient, DataType, Collection

class MilvusRAGSystem(OpenSourceRAGSystem):
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 milvus_uri: str = "./milvus_lite.db"):
        
        super().__init__(embedding_model_name)
        
        # Setup Milvus
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = "rag_documents"
        self.setup_milvus_collection()
    
    def setup_milvus_collection(self):
        """Setup Milvus collection for vector storage."""
        
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        
        # Create collection
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="IP",  # Inner Product
            consistency_level="Strong"
        )
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to Milvus collection."""
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
        
        # Prepare data for Milvus
        data = []
        for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
            data.append({
                "id": len(self.document_store) + i,
                "vector": emb.tolist(),
                "text": doc,
                "metadata": str(meta)  # Milvus requires string for JSON data
            })
        
        # Insert into Milvus
        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        # Update local storage
        self.document_store.extend(documents)
        self.metadata_store.extend(metadata)
    
    def search_documents(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search documents in Milvus."""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in Milvus
        search_results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_embedding.tolist(),
            limit=top_k,
            search_params={"metric_type": "IP", "params": {"nprobe": 10}}
        )
        
        results = []
        for result in search_results[0]:
            if result['distance'] >= threshold:
                results.append({
                    'document': result['entity']['text'],
                    'metadata': eval(result['entity']['metadata']),  # Convert back to dict
                    'similarity': result['distance'],
                    'index': result['entity']['id']
                })
        
        return results
```

## Advanced RAG Techniques

### Query Enhancement and Rewriting

```python
class AdvancedRAGSystem(OpenSourceRAGSystem):
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        super().__init__(embedding_model_name)
        
        # Initialize reranker
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(reranker_model_name)
        
        # Query enhancement model
        self.query_enhancer = self.setup_query_enhancer()
    
    def setup_query_enhancer(self):
        """Setup model for query enhancement."""
        
        # Use a smaller model for query enhancement
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        
        return {'tokenizer': tokenizer, 'model': model}
    
    def enhance_query(self, original_query: str) -> List[str]:
        """Generate enhanced versions of the query."""
        
        enhancement_prompts = [
            f"Rephrase this question: {original_query}",
            f"What are the key concepts in: {original_query}",
            f"Break down this question: {original_query}",
            f"Alternative phrasing: {original_query}"
        ]
        
        enhanced_queries = [original_query]  # Always include original
        
        for prompt in enhancement_prompts:
            try:
                inputs = self.query_enhancer['tokenizer'](
                    prompt, 
                    return_tensors="pt", 
                    max_length=100
                )
                
                with torch.no_grad():
                    outputs = self.query_enhancer['model'].generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 50,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.query_enhancer['tokenizer'].eos_token_id
                    )
                
                enhanced = self.query_enhancer['tokenizer'].decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                enhanced = enhanced[len(prompt):].strip()
                
                if enhanced and enhanced not in enhanced_queries:
                    enhanced_queries.append(enhanced)
                    
            except Exception as e:
                print(f"Query enhancement error: {e}")
                continue
        
        return enhanced_queries[:5]  # Return up to 5 variations
    
    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank retrieved documents using cross-encoder."""
        
        if not documents:
            return documents
        
        # Prepare query-document pairs
        pairs = [(query, doc['document']) for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Update documents with reranking scores
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            doc['combined_score'] = (doc['similarity'] + score) / 2
        
        # Sort by combined score
        reranked = sorted(documents, key=lambda x: x['combined_score'], reverse=True)
        
        return reranked
    
    def advanced_rag_query(self, query: str, top_k: int = 10, final_k: int = 5) -> Dict[str, Any]:
        """Execute advanced RAG with query enhancement and reranking."""
        
        # Enhance query
        enhanced_queries = self.enhance_query(query)
        
        # Retrieve documents for all query variations
        all_retrieved = []
        for enhanced_query in enhanced_queries:
            docs = self.search_documents(enhanced_query, top_k=top_k // len(enhanced_queries))
            all_retrieved.extend(docs)
        
        # Remove duplicates
        unique_docs = []
        seen_docs = set()
        for doc in all_retrieved:
            doc_hash = hash(doc['document'])
            if doc_hash not in seen_docs:
                unique_docs.append(doc)
                seen_docs.add(doc_hash)
        
        # Rerank documents
        reranked_docs = self.rerank_documents(query, unique_docs)[:final_k]
        
        # Generate response
        context_docs = [doc['document'] for doc in reranked_docs]
        answer = self.generate_response(query, context_docs)
        
        return {
            'query': query,
            'enhanced_queries': enhanced_queries,
            'answer': answer,
            'sources': reranked_docs,
            'retrieval_stats': {
                'total_retrieved': len(all_retrieved),
                'unique_docs': len(unique_docs),
                'final_docs': len(reranked_docs)
            }
        }
```

### Multi-Modal RAG

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

class MultiModalRAGSystem(AdvancedRAGSystem):
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        super().__init__(embedding_model_name)
        
        # Setup CLIP for image-text understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Storage for multimodal documents
        self.image_store = []
        self.image_embeddings = []
    
    def add_multimodal_documents(self, texts: List[str], images: List[str] = None,
                                metadata: List[Dict] = None):
        """Add documents with optional images."""
        
        # Add text documents as usual
        self.add_documents(texts, metadata)
        
        # Process images if provided
        if images:
            self.add_images(images, texts, metadata)
    
    def add_images(self, image_paths: List[str], associated_texts: List[str] = None,
                   metadata: List[Dict] = None):
        """Add images to the multimodal system."""
        
        if metadata is None:
            metadata = [{}] * len(image_paths)
        
        if associated_texts is None:
            associated_texts = [""] * len(image_paths)
        
        image_embeddings = []
        
        for img_path, text in zip(image_paths, associated_texts):
            try:
                # Load image
                if img_path.startswith('http'):
                    image = Image.open(requests.get(img_path, stream=True).raw)
                else:
                    image = Image.open(img_path)
                
                # Generate CLIP embedding
                inputs = self.clip_processor(
                    text=[text] if text else [""],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    image_embedding = outputs.image_embeds.cpu().numpy()
                
                image_embeddings.append(image_embedding[0])
                self.image_store.append(img_path)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        self.image_embeddings.extend(image_embeddings)
    
    def search_multimodal(self, query: str, include_images: bool = True,
                         top_k: int = 5) -> Dict[str, Any]:
        """Search across text and image modalities."""
        
        results = {'text': [], 'images': []}
        
        # Search text documents
        text_results = self.search_documents(query, top_k=top_k)
        results['text'] = text_results
        
        # Search images if requested
        if include_images and self.image_embeddings:
            image_results = self.search_images(query, top_k=top_k // 2)
            results['images'] = image_results
        
        return results
    
    def search_images(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search images using CLIP similarity."""
        
        if not self.image_embeddings:
            return []
        
        # Generate text embedding for query
        inputs = self.clip_processor(
            text=[query],
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            text_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy()
        
        # Calculate similarities
        similarities = []
        for img_emb in self.image_embeddings:
            similarity = np.dot(text_embedding[0], img_emb) / (
                np.linalg.norm(text_embedding[0]) * np.linalg.norm(img_emb)
            )
            similarities.append(similarity)
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.image_store[idx],
                'similarity': float(similarities[idx]),
                'index': idx
            })
        
        return results
```

## Agentic AI Systems

### Tool-Using Agents

```python
from transformers import Tool, ReactCodeAgent, HfEngine

class OpenSourceAgent:
    def __init__(self, llm_model_name: str = "microsoft/DialoGPT-medium"):
        self.llm_engine = HfEngine(llm_model_name)
        self.setup_tools()
        self.agent = self.create_agent()
    
    def setup_tools(self):
        """Setup tools for the agent."""
        
        @Tool
        def search_documents(query: str) -> str:
            """Search for relevant documents in the knowledge base."""
            # This would integrate with your RAG system
            rag_system = OpenSourceRAGSystem()
            result = rag_system.rag_query(query)
            return result['answer']
        
        @Tool
        def calculate(expression: str) -> str:
            """Perform mathematical calculations."""
            try:
                result = eval(expression)  # Note: Use safely in production
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        @Tool
        def web_search(query: str) -> str:
            """Search the web for information."""
            # Placeholder for web search integration
            return f"Web search results for: {query}"
        
        self.tools = [search_documents, calculate, web_search]
    
    def create_agent(self):
        """Create the reactive agent."""
        
        agent = ReactCodeAgent(
            tools=self.tools,
            llm_engine=self.llm_engine,
            max_iterations=10,
            verbose=True
        )
        
        return agent
    
    def run(self, user_input: str) -> str:
        """Run the agent with user input."""
        
        try:
            result = self.agent.run(user_input)
            return result
        except Exception as e:
            return f"Agent error: {e}"
```

### Multi-Agent Systems

```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.setup_specialized_agents()
    
    def setup_specialized_agents(self):
        """Setup specialized agents for different tasks."""
        
        # Research Agent
        class ResearchAgent(OpenSourceAgent):
            def __init__(self):
                super().__init__()
                self.specialty = "research"
            
            def research_topic(self, topic: str) -> Dict[str, Any]:
                """Conduct comprehensive research on a topic."""
                
                # Use RAG system for knowledge base search
                rag_results = self.search_documents(topic)
                
                # Structure findings
                return {
                    'topic': topic,
                    'findings': rag_results,
                    'sources': [],
                    'confidence': 0.8
                }
        
        # Code Agent
        class CodeAgent(OpenSourceAgent):
            def __init__(self):
                super().__init__()
                self.specialty = "coding"
            
            def generate_code(self, requirements: str) -> str:
                """Generate code based on requirements."""
                
                prompt = f"""
                Generate Python code for the following requirements:
                {requirements}
                
                Please provide clean, well-commented code:
                """
                
                return self.run(prompt)
        
        # Analysis Agent
        class AnalysisAgent(OpenSourceAgent):
            def __init__(self):
                super().__init__()
                self.specialty = "analysis"
            
            def analyze_data(self, data: str) -> Dict[str, Any]:
                """Analyze provided data."""
                
                analysis_prompt = f"""
                Analyze the following data and provide insights:
                {data}
                
                Include:
                - Key patterns
                - Trends
                - Recommendations
                """
                
                result = self.run(analysis_prompt)
                
                return {
                    'analysis': result,
                    'data_summary': data[:200] + "..." if len(data) > 200 else data,
                    'agent': 'analysis'
                }
        
        self.agents = {
            'research': ResearchAgent(),
            'code': CodeAgent(),
            'analysis': AnalysisAgent()
        }
    
    def route_request(self, user_input: str) -> str:
        """Route user request to appropriate agent."""
        
        # Simple routing based on keywords
        routing_keywords = {
            'research': ['research', 'find', 'search', 'information', 'about'],
            'code': ['code', 'program', 'function', 'implement', 'develop'],
            'analysis': ['analyze', 'insights', 'patterns', 'trends', 'data']
        }
        
        user_input_lower = user_input.lower()
        
        for agent_type, keywords in routing_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return self.agents[agent_type].run(user_input)
        
        # Default to research agent
        return self.agents['research'].run(user_input)
    
    def collaborative_task(self, task: str) -> Dict[str, Any]:
        """Execute task using multiple agents collaboratively."""
        
        results = {}
        
        # Research phase
        research_result = self.agents['research'].run(f"Research: {task}")
        results['research'] = research_result
        
        # Analysis phase
        analysis_result = self.agents['analysis'].analyze_data(research_result)
        results['analysis'] = analysis_result
        
        # Implementation phase (if applicable)
        if any(keyword in task.lower() for keyword in ['implement', 'code', 'build']):
            code_result = self.agents['code'].generate_code(task)
            results['code'] = code_result
        
        return results
```

## Production Deployment

### Model Serving and Scaling

```python
from flask import Flask, request, jsonify
import threading
import queue
import time

class ModelServer:
    def __init__(self, model_path: str, max_workers: int = 4):
        self.model_path = model_path
        self.max_workers = max_workers
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.workers = []
        
        self.setup_workers()
        self.setup_flask_app()
    
    def setup_workers(self):
        """Setup worker threads for model inference."""
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self.worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def worker_loop(self, worker_id: int):
        """Worker loop for processing inference requests."""
        
        # Load model in worker thread
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        while True:
            try:
                request_data = self.request_queue.get(timeout=1)
                
                # Process request
                prompt = request_data['prompt']
                request_id = request_data['id']
                max_length = request_data.get('max_length', 100)
                
                inputs = tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                # Put response in queue
                self.response_queue.put({
                    'id': request_id,
                    'response': response,
                    'worker_id': worker_id
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                self.response_queue.put({
                    'id': request_data.get('id', 'unknown'),
                    'error': str(e),
                    'worker_id': worker_id
                })
    
    def setup_flask_app(self):
        """Setup Flask application for serving."""
        
        self.app = Flask(__name__)
        
        @self.app.route('/generate', methods=['POST'])
        def generate():
            data = request.json
            prompt = data.get('prompt', '')
            max_length = data.get('max_length', 100)
            
            # Generate unique request ID
            request_id = str(time.time())
            
            # Add to request queue
            self.request_queue.put({
                'id': request_id,
                'prompt': prompt,
                'max_length': max_length
            })
            
            # Wait for response
            timeout = 30  # 30 second timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response_data = self.response_queue.get(timeout=1)
                    if response_data['id'] == request_id:
                        return jsonify(response_data)
                except queue.Empty:
                    continue
            
            return jsonify({'error': 'Request timeout'}), 408
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'workers': len(self.workers),
                'queue_size': self.request_queue.qsize()
            })
    
    def run(self, host: str = '0.0.0.0', port: int = 5000):
        """Run the model server."""
        self.app.run(host=host, port=port, threaded=True)
```

### Container Deployment

```python
# Dockerfile content for containerized deployment
dockerfile_content = '''
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Create cache directory
RUN mkdir -p /app/cache

# Run the application
CMD ["python", "model_server.py"]
'''

# Docker Compose for multi-service deployment
docker_compose_content = '''
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MODEL_PATH=/models/fine-tuned-model
      - MAX_WORKERS=4
    volumes:
      - ./models:/models
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  vector-db:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api

volumes:
  milvus_data:
'''
```

## Performance Optimization

### Memory Management and Quantization

```python
class OptimizedInference:
    def __init__(self, model_path: str, optimization_level: str = "balanced"):
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.setup_optimized_model()
    
    def setup_optimized_model(self):
        """Setup model with optimizations based on level."""
        
        optimization_configs = {
            "speed": {
                "torch_dtype": torch.float16,
                "quantization": "8bit",
                "device_map": "auto",
                "use_cache": True
            },
            "balanced": {
                "torch_dtype": torch.bfloat16,
                "quantization": "4bit",
                "device_map": "auto",
                "use_cache": True
            },
            "memory": {
                "torch_dtype": torch.float16,
                "quantization": "4bit",
                "device_map": "sequential",
                "use_cache": False
            }
        }
        
        config = optimization_configs[self.optimization_level]
        
        # Setup quantization config
        if config["quantization"] == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config["torch_dtype"],
                bnb_4bit_use_double_quant=True
            )
        elif config["quantization"] == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        
        # Load optimized model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            torch_dtype=config["torch_dtype"],
            device_map=config["device_map"],
            use_cache=config["use_cache"]
        )
        
        # Enable optimizations
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    def optimized_generate(self, prompt: str, **kwargs) -> str:
        """Generate text with optimizations."""
        
        # Default generation parameters for efficiency
        generation_config = {
            "max_length": kwargs.get("max_length", 200),
            "temperature": kwargs.get("temperature", 0.7),
            "do_sample": kwargs.get("do_sample", True),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # Generate with optimizations
        with torch.no_grad():
            if self.optimization_level == "speed":
                # Use torch.compile for speed (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    compiled_model = torch.compile(self.model)
                    outputs = compiled_model.generate(**inputs, **generation_config)
                else:
                    outputs = self.model.generate(**inputs, **generation_config)
            else:
                outputs = self.model.generate(**inputs, **generation_config)
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def batch_generate(self, prompts: List[str], batch_size: int = 4) -> List[str]:
        """Generate responses for multiple prompts efficiently."""
        
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            # Generate batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode batch responses
            batch_responses = []
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                original_prompt = batch_prompts[j]
                response = response[len(original_prompt):].strip()
                batch_responses.append(response)
            
            responses.extend(batch_responses)
        
        return responses
```

### Caching and Performance Monitoring

```python
import redis
import json
import hashlib
from datetime import datetime
import psutil

class PerformanceOptimizer:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True
        )
        self.cache_ttl = 3600  # 1 hour
        self.performance_metrics = {}
    
    def cached_inference(self, prompt: str, model_func, **kwargs) -> Dict[str, Any]:
        """Execute inference with caching."""
        
        # Create cache key
        cache_data = {"prompt": prompt, "kwargs": kwargs}
        cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
        
        # Check cache
        cached_result = self.redis_client.get(f"inference:{cache_key}")
        if cached_result:
            result = json.loads(cached_result)
            result['cache_hit'] = True
            return result
        
        # Execute inference
        start_time = time.time()
        response = model_func(prompt, **kwargs)
        inference_time = time.time() - start_time
        
        # Prepare result
        result = {
            'response': response,
            'inference_time': inference_time,
            'cache_hit': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        self.redis_client.setex(
            f"inference:{cache_key}",
            self.cache_ttl,
            json.dumps(result)
        )
        
        # Update performance metrics
        self.update_performance_metrics(inference_time)
        
        return result
    
    def update_performance_metrics(self, inference_time: float):
        """Update performance tracking metrics."""
        
        current_time = datetime.now()
        
        # System metrics
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory': self.get_gpu_memory() if torch.cuda.is_available() else None
        }
        
        # Inference metrics
        if 'inference_times' not in self.performance_metrics:
            self.performance_metrics['inference_times'] = []
        
        self.performance_metrics['inference_times'].append(inference_time)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics['inference_times']) > 100:
            self.performance_metrics['inference_times'] = \
                self.performance_metrics['inference_times'][-100:]
        
        # Store in Redis
        self.redis_client.setex(
            "performance_metrics",
            3600,
            json.dumps({
                'system': system_metrics,
                'avg_inference_time': np.mean(self.performance_metrics['inference_times']),
                'timestamp': current_time.isoformat()
            })
        )
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage."""
        
        if not torch.cuda.is_available():
            return None
        
        gpu_memory = {}
        for i in range(torch.cuda.device_count()):
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            
            gpu_memory[f'gpu_{i}'] = {
                'reserved_gb': memory_reserved,
                'allocated_gb': memory_allocated
            }
        
        return gpu_memory
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        cached_metrics = self.redis_client.get("performance_metrics")
        if cached_metrics:
            metrics = json.loads(cached_metrics)
        else:
            metrics = {}
        
        # Cache statistics
        cache_stats = {
            'total_keys': len(self.redis_client.keys("inference:*")),
            'cache_memory_usage': self.redis_client.memory_usage("performance_metrics") if self.redis_client.exists("performance_metrics") else 0
        }
        
        return {
            'system_metrics': metrics.get('system', {}),
            'inference_performance': {
                'avg_time': metrics.get('avg_inference_time', 0),
                'total_inferences': len(self.performance_metrics.get('inference_times', []))
            },
            'cache_performance': cache_stats,
            'timestamp': datetime.now().isoformat()
        }
```

## Best Practices Summary

### Open-Source Advantages

1. **Cost Effectiveness**: No API fees, full control over deployment costs
2. **Privacy and Security**: Complete data control, no external API calls
3. **Customization**: Full model fine-tuning and adaptation capabilities
4. **Transparency**: Open-source models provide full visibility into capabilities
5. **Community Support**: Extensive ecosystem and community contributions

### Implementation Guidelines

1. **Model Selection**: Choose models based on specific requirements and hardware constraints
2. **Quantization**: Use 4-bit or 8-bit quantization for memory efficiency
3. **Fine-Tuning**: Leverage LoRA and other PEFT methods for adaptation
4. **Caching**: Implement intelligent caching for repeated inference
5. **Monitoring**: Track performance metrics and system resources

### Production Considerations

1. **Scaling**: Implement worker-based serving for concurrent requests
2. **Containerization**: Use Docker for consistent deployment environments
3. **Load Balancing**: Distribute requests across multiple model instances
4. **Backup and Recovery**: Implement model versioning and backup strategies
5. **Security**: Secure API endpoints and implement proper authentication

This comprehensive guide provides a complete framework for building production-ready AI applications using the Hugging Face ecosystem, offering powerful alternatives to proprietary AI services while maintaining full control over data, costs, and customization.