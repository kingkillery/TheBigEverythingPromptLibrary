#!/bin/bash
set -e

echo "🐳 Starting The Big Everything Prompt Library..."

# Check if we're in development mode
if [ "$DEBUG" = "true" ]; then
    echo "🛠️  Development mode enabled"
    export PYTHONDONTWRITEBYTECODE=1
fi

# Ensure cache directories exist
mkdir -p /app/web_interface/backend/embeddings_cache
mkdir -p /app/.cache/torch
mkdir -p /app/.cache/huggingface

# Download models if not cached (production only)
if [ "$DEBUG" != "true" ]; then
    echo "📦 Checking for pre-trained models..."
    python -c "
import os
try:
    from sentence_transformers import SentenceTransformer
    model_path = '/app/.cache/huggingface/sentence-transformers'
    if not os.path.exists(model_path):
        print('Downloading sentence-transformer model...')
        SentenceTransformer('all-MiniLM-L6-v2')
        print('✅ Model downloaded successfully')
    else:
        print('✅ Model already cached')
except Exception as e:
    print(f'⚠️  Model download failed: {e}')
"
fi

# Test the enhanced search functionality
echo "🧪 Testing enhanced search capabilities..."
python /app/web_interface/test_enhanced_search.py --quick || echo "⚠️  Some enhanced features may not be available"

# Start the application
echo "🚀 Starting web interface on port 8000..."
exec "$@"