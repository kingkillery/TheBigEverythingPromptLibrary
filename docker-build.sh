#!/bin/bash
# Quick Docker build and test script

set -e

echo "🐳 Building The Big Everything Prompt Library Docker Environment"
echo "================================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Check if Make is available
if command_exists make; then
    echo "✅ Make found - using Makefile commands"
    USE_MAKE=true
else
    echo "⚠️  Make not found - using direct docker-compose commands"
    USE_MAKE=false
fi

# Build the image
echo ""
echo "🔨 Building Docker image..."
if [ "$USE_MAKE" = true ]; then
    make build
else
    docker-compose build
fi

echo ""
echo "✅ Build completed successfully!"

# Offer to start the application
echo ""
read -p "🚀 Would you like to start the application now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting application..."
    
    if [ "$USE_MAKE" = true ]; then
        make run
    else
        docker-compose up -d
    fi
    
    echo ""
    echo "🎉 Application started successfully!"
    echo ""
    echo "📍 Access Points:"
    echo "   🌐 Web Interface: http://localhost:8000"
    echo "   📚 API Docs: http://localhost:8000/docs"
    echo "   🔍 Enhanced Search: All features enabled"
    echo ""
    echo "💡 Useful commands:"
    if [ "$USE_MAKE" = true ]; then
        echo "   make logs     # View application logs"
        echo "   make status   # Check service status"
        echo "   make stop     # Stop all services"
        echo "   make help     # Show all commands"
    else
        echo "   docker-compose logs -f    # View application logs"
        echo "   docker-compose ps         # Check service status"
        echo "   docker-compose down       # Stop all services"
    fi
    
    # Test the application
    echo ""
    echo "🧪 Testing application in 10 seconds..."
    sleep 10
    
    if curl -s http://localhost:8000/api/categories >/dev/null; then
        echo "✅ Application is responding correctly!"
    else
        echo "⚠️  Application may still be starting up. Check logs if issues persist."
    fi
else
    echo ""
    echo "📝 To start the application later:"
    if [ "$USE_MAKE" = true ]; then
        echo "   make run"
    else
        echo "   docker-compose up -d"
    fi
fi

echo ""
echo "🎉 Docker environment setup complete!"