# Fine-tuning a Vision Transformer Model With a Custom Biomedical Dataset

**Source:** [Hugging Face Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vit_custom_dataset)  
**Date Added:** June 10, 2025  
**Category:** Computer Vision, Medical AI, Fine-Tuning, Vision Transformers

## Overview

This comprehensive guide demonstrates how to fine-tune a Vision Transformer (ViT) model on a custom biomedical dataset for medical image classification. Learn to adapt Google's pre-trained ViT model for domain-specific medical imaging tasks with high accuracy and robust evaluation methods.

## Key Objectives

- **Custom Dataset Integration**: Load and prepare biomedical image datasets for training
- **Vision Transformer Fine-tuning**: Adapt pre-trained ViT models for medical image classification
- **Data Processing Pipeline**: Implement proper image preprocessing and augmentation techniques
- **Model Evaluation**: Use comprehensive metrics and visualization for performance assessment
- **Production Deployment**: Prepare models for real-world medical applications

## Why Vision Transformers for Medical Imaging?

**Advantages:**
- **Global Context**: ViTs capture long-range dependencies better than CNNs
- **Transfer Learning**: Pre-trained on large datasets (ImageNet) for excellent feature extraction
- **Scalability**: Performance improves with larger datasets and model sizes
- **Interpretability**: Attention mechanisms provide insights into model decisions

## Prerequisites

### Hardware Requirements
- GPU with CUDA support (T4 or better recommended)
- Minimum 8GB GPU memory
- 16GB+ RAM

### Software Requirements
```bash
# Core libraries
pip install datasets transformers accelerate torch torchvision
pip install scikit-learn matplotlib wandb
```

### Dataset Requirements
- Medical images in common formats (JPG, PNG, DICOM)
- Organized class structure
- Sufficient samples per class (minimum 50-100 per class)

## Dataset Information

This guide uses a custom breast cancer ultrasound dataset containing:
- **Total Images**: 780
- **Classes**: 3 (benign, malignant, normal)
- **Format**: Standard image files
- **Medical Domain**: Breast cancer screening

### Dataset Structure
```
dataset/
├── train/
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── validation/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── test/
    ├── benign/
    ├── malignant/
    └── normal/
```

## Model Information

**Base Model**: Google's `vit-large-patch16-224`
- **Pre-training**: ImageNet-21k (14M images, 21,843 classes)
- **Fine-tuning**: ImageNet 2012 (1M images, 1,000 classes)
- **Resolution**: 224×224 pixels
- **Architecture**: Large variant with 16×16 patches

## Step-by-Step Implementation

### 1. Environment Setup

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score
import os

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
```

### 2. Dataset Loading and Preparation

#### 2.1 Load Dataset
```python
# Option 1: Load from Hugging Face Hub
dataset = load_dataset("emre570/breastcancer-ultrasound-images")

# Option 2: Load from local directory
# from datasets import load_dataset
# dataset = load_dataset("imagefolder", data_dir="path/to/your/dataset")

print("Dataset structure:")
print(dataset)
```

#### 2.2 Create Validation Split
```python
# Calculate validation split size
test_num = len(dataset["test"])
train_num = len(dataset["train"])
val_size = test_num / train_num

# Split training data
train_val_split = dataset["train"].train_test_split(test_size=val_size, stratify_by_column="label")

# Create final dataset structure
dataset = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": dataset["test"]
})

print(f"Final dataset structure:")
print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")
print(f"Test: {len(dataset['test'])} samples")

# Assign for easy reference
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]
```

#### 2.3 Visualize Sample Data
```python
def visualize_samples(dataset, num_samples=3):
    """Visualize one sample from each class"""
    plt.figure(figsize=(15, 5))
    
    shown_labels = set()
    sample_idx = 0
    
    for i, sample in enumerate(dataset):
        label = dataset.features["label"].names[sample["label"]]
        
        if label not in shown_labels and len(shown_labels) < num_samples:
            plt.subplot(1, num_samples, len(shown_labels) + 1)
            plt.imshow(sample["image"])
            plt.title(f"{label}\n(Class {sample['label']})")
            plt.axis("off")
            shown_labels.add(label)
    
    plt.tight_layout()
    plt.show()

# Visualize samples
visualize_samples(train_ds)
```

### 3. Data Processing Pipeline

#### 3.1 Label Mapping
```python
# Create label mappings
id2label = {id: label for id, label in enumerate(train_ds.features["label"].names)}
label2id = {label: id for id, label in id2label.items()}

print("Label mappings:")
print(f"ID to Label: {id2label}")
print(f"Label to ID: {label2id}")
```

#### 3.2 Image Processor Setup
```python
# Initialize the ViT image processor
model_name = "google/vit-large-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)

print(f"Image processor configuration:")
print(f"Image mean: {processor.image_mean}")
print(f"Image std: {processor.image_std}")
print(f"Size: {processor.size}")
```

#### 3.3 Data Transformations
```python
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    ColorJitter,
    ToTensor,
    Resize,
)

# Get processor parameters
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Normalization transform
normalize = Normalize(mean=image_mean, std=image_std)

# Training transforms (with augmentation)
train_transforms = Compose([
    RandomResizedCrop(size, scale=(0.8, 1.0)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    normalize,
])

# Validation/Test transforms (no augmentation)
val_transforms = Compose([
    Resize((size, size)),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

test_transforms = Compose([
    Resize((size, size)),
    CenterCrop(size),
    ToTensor(),
    normalize,
])
```

#### 3.4 Transform Functions
```python
def apply_train_transforms(examples):
    """Apply training transforms to examples"""
    examples["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

def apply_val_transforms(examples):
    """Apply validation transforms to examples"""
    examples["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

def apply_test_transforms(examples):
    """Apply test transforms to examples"""
    examples["pixel_values"] = [
        test_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

# Apply transforms to datasets
train_ds.set_transform(apply_train_transforms)
val_ds.set_transform(apply_val_transforms)
test_ds.set_transform(apply_test_transforms)

print("Transforms applied successfully!")
```

### 4. Data Loading Setup

#### 4.1 Custom Collate Function
```python
def collate_fn(examples):
    """Custom collate function for batching"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Create data loaders
train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=4, shuffle=False)
test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size=4, shuffle=False)
```

#### 4.2 Verify Data Loading
```python
# Test batch loading
batch = next(iter(train_dl))
print("Batch verification:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")

# Expected output:
# pixel_values: torch.Size([4, 3, 224, 224])
# labels: torch.Size([4])
```

### 5. Model Configuration and Training

#### 5.1 Model Initialization
```python
# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # Important for different number of classes
)

print(f"Model loaded: {model_name}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

#### 5.2 Training Configuration
```python
from transformers import TrainingArguments, Trainer
import wandb

# Optional: Initialize Weights & Biases for logging
# wandb.init(project="vit-medical-classification")

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-medical-classifier",
    save_total_limit=2,
    report_to="wandb",  # Remove if not using W&B
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=40,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_accumulation_steps=2,  # Effective batch size = 20
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,  # Mixed precision training
)
```

#### 5.3 Metrics and Evaluation
```python
def compute_metrics(eval_pred):
    """Compute accuracy and other metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": accuracy,
    }
```

#### 5.4 Training Setup
```python
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

print("Trainer initialized successfully!")
```

#### 5.5 Model Training
```python
# Start training
print("Starting training...")
train_result = trainer.train()

print("Training completed!")
print(f"Training results: {train_result}")

# Save the final model
trainer.save_model("./final-vit-medical-model")
processor.save_pretrained("./final-vit-medical-model")
```

### 6. Model Evaluation

#### 6.1 Test Set Evaluation
```python
# Evaluate on test set
print("Evaluating on test set...")
outputs = trainer.predict(test_ds)

print("Test results:")
print(f"Test Loss: {outputs.metrics['test_loss']:.4f}")
print(f"Test Accuracy: {outputs.metrics.get('test_accuracy', 'N/A')}")
print(f"Test Runtime: {outputs.metrics['test_runtime']:.2f} seconds")
print(f"Samples per second: {outputs.metrics['test_samples_per_second']:.2f}")
```

#### 6.2 Detailed Performance Analysis
```python
# Extract predictions and true labels
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

# Calculate recall scores
recall_scores = recall_score(y_true, y_pred, average=None)
labels = train_ds.features["label"].names

print("\nRecall Scores by Class:")
for label, score in zip(labels, recall_scores):
    print(f"Recall for {label}: {score:.3f}")

# Calculate overall metrics
overall_accuracy = (y_pred == y_true).mean()
macro_recall = recall_score(y_true, y_pred, average='macro')
weighted_recall = recall_score(y_true, y_pred, average='weighted')

print(f"\nOverall Metrics:")
print(f"Accuracy: {overall_accuracy:.3f}")
print(f"Macro Recall: {macro_recall:.3f}")
print(f"Weighted Recall: {weighted_recall:.3f}")
```

### 7. Results Visualization

#### 7.1 Confusion Matrix
```python
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """Plot confusion matrix with labels"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    plt.figure(figsize=(10, 8))
    disp.plot(xticks_rotation=45, cmap='Blues', values_format='d')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return cm

# Plot confusion matrix
cm = plot_confusion_matrix(y_true, y_pred, labels, "ViT Medical Classification Results")

# Print confusion matrix details
print("\nConfusion Matrix Analysis:")
for i, label in enumerate(labels):
    true_positives = cm[i, i]
    false_positives = cm[:, i].sum() - true_positives
    false_negatives = cm[i, :].sum() - true_positives
    true_negatives = cm.sum() - true_positives - false_positives - false_negatives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{label}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
```

#### 7.2 Training History Visualization
```python
def plot_training_history(trainer):
    """Plot training and validation metrics"""
    logs = trainer.state.log_history
    
    train_loss = [log['train_loss'] for log in logs if 'train_loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
    eval_accuracy = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]
    
    epochs = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    if eval_loss:
        ax1.plot(epochs[:len(eval_loss)], eval_loss, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    if eval_accuracy:
        ax2.plot(epochs[:len(eval_accuracy)], eval_accuracy, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(trainer)
```

### 8. Model Inference and Prediction

#### 8.1 Single Image Prediction
```python
def predict_single_image(model, processor, image_path, id2label):
    """Predict class for a single image"""
    from PIL import Image
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class_id = predictions.argmax().item()
    confidence = predictions[0][predicted_class_id].item()
    predicted_label = id2label[predicted_class_id]
    
    return predicted_label, confidence, predictions[0].tolist()

# Example usage
# predicted_label, confidence, all_probs = predict_single_image(
#     model, processor, "path/to/test/image.jpg", id2label
# )
# print(f"Predicted: {predicted_label} (Confidence: {confidence:.3f})")
```

#### 8.2 Batch Prediction
```python
def predict_batch(model, dataloader, id2label):
    """Predict classes for a batch of images"""
    model.eval()
    predictions = []
    true_labels = []
    confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(pixel_values=batch["pixel_values"])
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            batch_preds = torch.argmax(probs, dim=-1)
            batch_confs = torch.max(probs, dim=-1)[0]
            
            predictions.extend(batch_preds.cpu().numpy())
            confidences.extend(batch_confs.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())
    
    return predictions, true_labels, confidences

# Get predictions for test set
test_predictions, test_true, test_confidences = predict_batch(model, test_dl, id2label)

# Analyze confidence distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(test_confidences, bins=20, alpha=0.7, edgecolor='black')
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
correct_mask = np.array(test_predictions) == np.array(test_true)
plt.hist(np.array(test_confidences)[correct_mask], bins=20, alpha=0.7, label='Correct', edgecolor='black')
plt.hist(np.array(test_confidences)[~correct_mask], bins=20, alpha=0.7, label='Incorrect', edgecolor='black')
plt.title('Confidence by Correctness')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
```

### 9. Model Deployment Preparation

#### 9.1 Model Optimization
```python
# Convert to TorchScript for deployment
def convert_to_torchscript(model, processor, save_path):
    """Convert model to TorchScript for deployment"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Trace the model
    traced_model = torch.jit.trace(model.vit, dummy_input)
    
    # Save traced model
    traced_model.save(f"{save_path}/traced_model.pt")
    
    print(f"TorchScript model saved to {save_path}/traced_model.pt")

# Convert model
convert_to_torchscript(model, processor, "./final-vit-medical-model")
```

#### 9.2 Model Quantization
```python
def quantize_model(model, save_path):
    """Apply dynamic quantization for smaller model size"""
    model.eval()
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), f"{save_path}/quantized_model.pth")
    
    print(f"Quantized model saved to {save_path}/quantized_model.pth")

# Quantize model
quantize_model(model, "./final-vit-medical-model")
```

#### 9.3 Create Inference API
```python
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

def create_inference_api(model, processor, id2label):
    """Create a simple REST API for model inference"""
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get image from request
            data = request.json
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Make prediction
            inputs = processor(images=image, return_tensors="pt")
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get results
            predicted_class_id = predictions.argmax().item()
            confidence = predictions[0][predicted_class_id].item()
            predicted_label = id2label[predicted_class_id]
            
            return jsonify({
                'prediction': predicted_label,
                'confidence': float(confidence),
                'class_probabilities': {
                    id2label[i]: float(predictions[0][i]) 
                    for i in range(len(id2label))
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy'})
    
    return app

# Create API
# api = create_inference_api(model, processor, id2label)
# api.run(host='0.0.0.0', port=5000)
```

### 10. Advanced Techniques

#### 10.1 Attention Visualization
```python
def visualize_attention(model, processor, image_path, save_path=None):
    """Visualize attention maps for interpretability"""
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # List of attention weights for each layer
    
    # Get the last layer attention weights
    last_attention = attentions[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]
    
    # Average over heads and remove CLS token
    attention_map = last_attention[0].mean(dim=0)[1:, 1:]  # Remove CLS token
    
    # Reshape to spatial dimensions (14x14 for ViT-224)
    grid_size = int(np.sqrt(attention_map.shape[0]))
    attention_map = attention_map.reshape(grid_size, grid_size)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attention_map.detach().numpy(), cmap='hot', interpolation='nearest')
    axes[1].set_title("Attention Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
# visualize_attention(model, processor, "path/to/image.jpg", "attention_map.png")
```

#### 10.2 Gradual Unfreezing Strategy
```python
def gradual_unfreezing_training(model, trainer, num_epochs_per_stage=5):
    """Implement gradual unfreezing for better fine-tuning"""
    
    # Stage 1: Only classifier head
    for param in model.vit.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    print("Stage 1: Training classifier head only...")
    trainer.args.num_train_epochs = num_epochs_per_stage
    trainer.train()
    
    # Stage 2: Unfreeze last encoder layer
    for param in model.vit.encoder.layer[-1].parameters():
        param.requires_grad = True
    
    print("Stage 2: Training with last encoder layer...")
    trainer.args.num_train_epochs = num_epochs_per_stage
    trainer.train()
    
    # Stage 3: Unfreeze all layers with lower learning rate
    for param in model.parameters():
        param.requires_grad = True
    
    # Reduce learning rate
    trainer.args.learning_rate = trainer.args.learning_rate * 0.1
    
    print("Stage 3: Fine-tuning all layers...")
    trainer.args.num_train_epochs = num_epochs_per_stage
    trainer.train()

# Example usage
# gradual_unfreezing_training(model, trainer, num_epochs_per_stage=10)
```

#### 10.3 Class Imbalance Handling
```python
def handle_class_imbalance(train_dataset, strategy='weighted_loss'):
    """Handle class imbalance in the dataset"""
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    
    # Count samples per class
    labels = [sample['label'] for sample in train_dataset]
    class_counts = Counter(labels)
    
    print("Class distribution:")
    for label_id, count in class_counts.items():
        label_name = train_dataset.features["label"].names[label_id]
        print(f"{label_name}: {count} samples")
    
    if strategy == 'weighted_sampler':
        # Create weighted sampler
        total_samples = len(labels)
        class_weights = {label: total_samples / count for label, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler
    
    elif strategy == 'weighted_loss':
        # Calculate class weights for loss function
        total_samples = len(labels)
        class_weights = []
        for i in range(len(class_counts)):
            weight = total_samples / (len(class_counts) * class_counts[i])
            class_weights.append(weight)
        
        return torch.FloatTensor(class_weights)

# Example usage
class_weights = handle_class_imbalance(train_ds, strategy='weighted_loss')

# Use in loss function
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
```

### 11. Performance Monitoring

#### 11.1 Real-time Monitoring Dashboard
```python
import wandb
from datetime import datetime

def setup_monitoring(project_name="vit-medical-monitoring"):
    """Setup comprehensive monitoring"""
    
    config = {
        "model_name": "vit-large-patch16-224",
        "dataset": "breast-cancer-ultrasound",
        "num_classes": 3,
        "image_size": 224,
        "batch_size": 10,
        "learning_rate": 2e-5,
        "num_epochs": 40,
        "timestamp": datetime.now().isoformat()
    }
    
    wandb.init(project=project_name, config=config)
    
    return config

def log_model_metrics(trainer, test_results, epoch):
    """Log comprehensive metrics"""
    metrics = {
        "epoch": epoch,
        "test_accuracy": test_results["test_accuracy"],
        "test_loss": test_results["test_loss"],
        "learning_rate": trainer.get_last_lr()[0],
        "model_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    }
    
    wandb.log(metrics)

# Setup monitoring
# config = setup_monitoring()
```

#### 11.2 Model Performance Tracking
```python
def track_model_performance(model, test_loader, device='cuda'):
    """Track inference speed and memory usage"""
    model.eval()
    
    # Warm up
    dummy_batch = next(iter(test_loader))
    with torch.no_grad():
        _ = model(pixel_values=dummy_batch["pixel_values"].to(device))
    
    # Track inference time
    import time
    start_time = time.time()
    
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            _ = model(pixel_values=pixel_values)
            total_samples += pixel_values.shape[0]
    
    end_time = time.time()
    
    inference_time = end_time - start_time
    samples_per_second = total_samples / inference_time
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"GPU Memory Used: {memory_used:.2f} GB")
    
    print(f"Inference Performance:")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {inference_time:.2f} seconds")
    print(f"Samples per second: {samples_per_second:.2f}")
    print(f"Average time per sample: {inference_time/total_samples*1000:.2f} ms")

# Track performance
# track_model_performance(model, test_dl)
```

## Best Practices

### 1. Data Quality
- **Consistent Preprocessing**: Ensure all images follow the same preprocessing pipeline
- **Quality Control**: Manually review samples to ensure correct labeling
- **Data Augmentation**: Use medical-appropriate augmentations (avoid unrealistic transformations)

### 2. Model Training
- **Learning Rate Scheduling**: Use cosine annealing or step decay for better convergence
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Gradient Clipping**: Prevent exploding gradients with `max_grad_norm=1.0`

### 3. Evaluation Strategy
- **Stratified Splits**: Ensure balanced representation across train/val/test sets
- **Multiple Metrics**: Use accuracy, precision, recall, F1-score, and AUC
- **Cross-Validation**: Consider k-fold CV for smaller datasets

### 4. Medical AI Considerations
- **Regulatory Compliance**: Follow FDA/EMA guidelines for medical AI
- **Bias Testing**: Evaluate performance across different demographics
- **Uncertainty Quantification**: Implement confidence estimation for clinical decisions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   per_device_train_batch_size = 4
   per_device_eval_batch_size = 2
   
   # Enable gradient accumulation
   gradient_accumulation_steps = 4
   
   # Use mixed precision
   fp16 = True
   ```

2. **Poor Performance**
   ```python
   # Try different learning rates
   learning_rate = [1e-5, 2e-5, 5e-5, 1e-4]
   
   # Adjust warmup
   warmup_ratio = 0.1
   
   # Check data quality and preprocessing
   ```

3. **Slow Training**
   ```python
   # Enable DataLoader optimizations
   dataloader_pin_memory = True
   dataloader_num_workers = 4
   
   # Use compiled model (PyTorch 2.0+)
   model = torch.compile(model)
   ```

## Deployment Considerations

### 1. Model Serving
```python
# FastAPI deployment example
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Load model once at startup
model = ViTForImageClassification.from_pretrained("./final-vit-medical-model")
processor = ViTImageProcessor.from_pretrained("./final-vit-medical-model")
model.eval()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read and process image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Make prediction
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Return results
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = torch.max(predictions).item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "all_probabilities": predictions[0].tolist()
    }
```

### 2. Docker Containerization
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Test Accuracy | 91.03% | Overall classification accuracy |
| Model Size | ~300MB | Full precision model |
| Inference Speed | ~50ms | Per image on T4 GPU |
| Training Time | 2-3 hours | 40 epochs on T4 GPU |
| GPU Memory | ~6GB | During training |

### Class-Specific Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|---------|----------|
| Benign | 0.92 | 0.90 | 0.91 |
| Malignant | 0.89 | 0.86 | 0.87 |
| Normal | 0.85 | 0.78 | 0.81 |

## Conclusion

This comprehensive guide demonstrates how to successfully fine-tune Vision Transformers for medical image classification. The approach combines:

- **Modern Architecture**: ViT's attention mechanisms for better feature learning
- **Transfer Learning**: Leveraging pre-trained weights for faster convergence
- **Robust Evaluation**: Comprehensive metrics and visualization for thorough assessment
- **Production Ready**: Deployment considerations and optimization techniques

The methodology can be adapted for various medical imaging tasks including:
- Radiological diagnosis
- Pathology slide analysis
- Dermatological assessment
- Ophthalmological screening

## Next Steps

1. **Dataset Expansion**: Increase dataset size and diversity for better generalization
2. **Multi-Modal Integration**: Combine imaging with clinical metadata
3. **Uncertainty Quantification**: Implement Bayesian approaches for confidence estimation
4. **Federated Learning**: Enable collaborative training across institutions
5. **Real-Time Processing**: Optimize for edge deployment in medical devices

## Resources

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Medical AI Guidelines](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)
- [ViT Fine-tuning Best Practices](https://huggingface.co/blog/fine-tune-vit)

---

*This guide is part of TheBigEverythingPromptLibrary - your comprehensive resource for AI prompts and techniques.*