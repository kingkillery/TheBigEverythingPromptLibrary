# Fine-Tune a Chatbot on Your Documentation with Synthetic Data

**Source:** [Hugging Face Cookbook](https://huggingface.co/learn/cookbook/en/fine_tune_chatbot_docs_synthetic)  
**Date Added:** June 10, 2025  
**Category:** AI Training, Fine-Tuning, Synthetic Data

## Overview

This comprehensive guide demonstrates how to create a domain-specific Question & Answering chatbot using synthetic data generation and efficient fine-tuning techniques. Learn to transform your documentation into an intelligent assistant that can answer questions about your specific domain.

## Key Objectives

- Generate high-quality synthetic Q&A pairs from existing documentation
- Fine-tune a small, efficient Language Model for domain-specific tasks
- Create a specialized chatbot that understands your documentation
- Use minimal computational resources through efficient training techniques

## Why This Approach?

**Benefits:**
- **Cost-effective**: No need for manual annotation of Q&A pairs
- **Scalable**: Generate thousands of training examples automatically
- **Domain-specific**: Tailored responses for your specific use case
- **Resource-efficient**: Use smaller models with LoRA fine-tuning

## Prerequisites

### Hardware Requirements
- GPU with CUDA support (T4 or better recommended)
- Minimum 16GB RAM
- CUDA-enabled environment

### Software Requirements
```bash
# Core libraries
pip install unsloth vllm synthetic-data-kit
pip install transformers trl datasets
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step-by-Step Implementation

### 1. Environment Setup

```python
import os
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from synthetic_data_kit import SyntheticDataGenerator

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### 2. Data Preparation

#### 2.1 Documentation Processing

```python
def load_documentation(file_path):
    """Load and preprocess documentation text"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split into chunks for better processing
    chunks = content.split('\n\n')
    # Filter out very short chunks
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
    
    return chunks

# Load your documentation
doc_chunks = load_documentation('your_documentation.txt')
print(f"Loaded {len(doc_chunks)} documentation chunks")
```

#### 2.2 Synthetic Q&A Generation

```python
# Initialize synthetic data generator
generator = SyntheticDataGenerator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    temperature=0.7,
    max_tokens=512
)

def generate_qa_pairs(chunks, num_pairs_per_chunk=2):
    """Generate Q&A pairs from documentation chunks"""
    qa_pairs = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Generate questions and answers
        pairs = generator.generate_qa_pairs(
            context=chunk,
            num_pairs=num_pairs_per_chunk,
            question_types=['factual', 'conceptual', 'procedural']
        )
        
        qa_pairs.extend(pairs)
    
    return qa_pairs

# Generate synthetic data
synthetic_data = generate_qa_pairs(doc_chunks[:50])  # Start with subset
print(f"Generated {len(synthetic_data)} Q&A pairs")
```

### 3. Data Formatting

```python
def format_training_data(qa_pairs):
    """Format Q&A pairs for training"""
    formatted_data = []
    
    for pair in qa_pairs:
        # Create conversation format
        conversation = {
            "messages": [
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]}
            ]
        }
        formatted_data.append(conversation)
    
    return formatted_data

# Format data for training
training_data = format_training_data(synthetic_data)

# Create dataset
dataset = Dataset.from_list(training_data)
print(f"Created dataset with {len(dataset)} examples")
```

### 4. Model Setup and Fine-Tuning

#### 4.1 Load Base Model

```python
# Model configuration
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

#### 4.2 Add LoRA Adapters

```python
# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

#### 4.3 Training Configuration

```python
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=500,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="no",
    dataloader_pin_memory=False,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)
```

### 5. Training Execution

```python
# Start training
print("Starting training...")
trainer_stats = trainer.train()

# Save the fine-tuned model
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_model")

print("Training completed!")
print(f"Training stats: {trainer_stats}")
```

### 6. Inference and Testing

```python
def ask_chatbot(question, max_new_tokens=256):
    """Generate response to a question"""
    # Format the input
    messages = [
        {"role": "user", "content": question}
    ]
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response.strip()

# Test the chatbot
test_questions = [
    "What is the main purpose of this documentation?",
    "How do I get started with the basic features?",
    "What are the system requirements?",
]

for question in test_questions:
    print(f"Q: {question}")
    print(f"A: {ask_chatbot(question)}")
    print("-" * 50)
```

## Advanced Techniques

### 1. Data Quality Improvement

```python
def filter_qa_pairs(qa_pairs, min_question_length=10, min_answer_length=20):
    """Filter Q&A pairs based on quality criteria"""
    filtered_pairs = []
    
    for pair in qa_pairs:
        question = pair["question"].strip()
        answer = pair["answer"].strip()
        
        # Quality checks
        if (len(question) >= min_question_length and 
            len(answer) >= min_answer_length and
            question.endswith('?') and
            not answer.lower().startswith('i don\'t know')):
            filtered_pairs.append(pair)
    
    return filtered_pairs

# Apply filtering
high_quality_data = filter_qa_pairs(synthetic_data)
print(f"Filtered to {len(high_quality_data)} high-quality pairs")
```

### 2. Multi-Turn Conversations

```python
def create_multi_turn_conversations(qa_pairs):
    """Create multi-turn conversations from Q&A pairs"""
    conversations = []
    
    for i in range(0, len(qa_pairs), 3):  # Group every 3 Q&A pairs
        conversation = {
            "messages": []
        }
        
        for j in range(min(3, len(qa_pairs) - i)):
            pair = qa_pairs[i + j]
            conversation["messages"].extend([
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]}
            ])
        
        conversations.append(conversation)
    
    return conversations

# Create multi-turn data
multi_turn_data = create_multi_turn_conversations(high_quality_data)
```

### 3. Evaluation Framework

```python
def evaluate_chatbot(test_questions, reference_answers):
    """Evaluate chatbot performance"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Load sentence transformer for similarity
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    similarities = []
    
    for question, reference in zip(test_questions, reference_answers):
        # Generate chatbot response
        generated = ask_chatbot(question)
        
        # Calculate similarity
        embeddings = similarity_model.encode([reference, generated])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        similarities.append(similarity)
        
        print(f"Question: {question}")
        print(f"Reference: {reference}")
        print(f"Generated: {generated}")
        print(f"Similarity: {similarity:.3f}")
        print("-" * 80)
    
    return np.mean(similarities)

# Run evaluation
avg_similarity = evaluate_chatbot(test_questions, reference_answers)
print(f"Average similarity score: {avg_similarity:.3f}")
```

## Best Practices

### 1. Data Quality
- **Diverse Questions**: Generate different types of questions (factual, conceptual, procedural)
- **Context Relevance**: Ensure questions are directly answerable from the provided context
- **Quality Filtering**: Remove low-quality or nonsensical Q&A pairs

### 2. Training Optimization
- **Gradual Training**: Start with a subset of data, then scale up
- **Learning Rate Scheduling**: Use warmup and decay for stable training
- **Gradient Accumulation**: Handle memory constraints with smaller batch sizes

### 3. Model Selection
- **Size vs Performance**: Balance model size with computational resources
- **Domain Alignment**: Choose base models aligned with your domain
- **Fine-tuning Strategy**: Use LoRA for efficient adaptation

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   # Reduce batch size and increase gradient accumulation
   per_device_train_batch_size=1
   gradient_accumulation_steps=8
   ```

2. **Poor Quality Responses**
   ```python
   # Improve data filtering and increase training epochs
   num_train_epochs=5
   learning_rate=1e-4
   ```

3. **Slow Training**
   ```python
   # Use gradient checkpointing and optimize data loading
   use_gradient_checkpointing="unsloth"
   dataloader_num_workers=4
   ```

## Deployment Considerations

### 1. Model Optimization

```python
# Convert to GGUF format for efficient inference
from unsloth import FastLanguageModel

model.save_pretrained_gguf(
    "chatbot_model_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### 2. API Wrapper

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    response = ask_chatbot(question)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Training Time | ~2 hours | On T4 GPU with 500 steps |
| Model Size | ~1.5GB | 4-bit quantized with LoRA |
| Inference Speed | ~2 tokens/sec | On T4 GPU |
| Memory Usage | ~6GB | During training |

## Conclusion

This guide provides a complete pipeline for creating domain-specific chatbots using synthetic data and efficient fine-tuning. The approach is:

- **Scalable**: Generate training data automatically
- **Cost-effective**: Use smaller models and efficient training
- **Customizable**: Adapt to any domain or documentation
- **Production-ready**: Includes deployment considerations

## Next Steps

1. **Scale Up**: Increase documentation size and synthetic data volume
2. **Multi-domain**: Combine multiple documentation sources
3. **Advanced Evaluation**: Implement comprehensive evaluation metrics
4. **Production Deploy**: Set up robust inference infrastructure

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Synthetic Data Kit](https://github.com/synthetic-data-kit)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

*This guide is part of TheBigEverythingPromptLibrary - your comprehensive resource for AI prompts and techniques.*