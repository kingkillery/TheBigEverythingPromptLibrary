#!/bin/bash

echo "🚀 Starting The Big Everything Prompt Library Web Interface..."
echo "📍 Server will be available at: http://localhost:8000"
echo "🔍 API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to backend directory
cd "$SCRIPT_DIR/backend"

# Start the server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload