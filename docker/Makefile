# Makefile for The Big Everything Prompt Library Docker Environment

.PHONY: help build run stop clean dev prod logs test shell

# Default target
help: ## Show this help message
	@echo "🐳 The Big Everything Prompt Library - Docker Commands"
	@echo "=================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build the Docker image
	@echo "🔨 Building Docker image..."
	docker-compose build

build-dev: ## Build development image
	@echo "🔨 Building development Docker image..."
	docker-compose -f docker-compose.dev.yml build

# Run targets
run: ## Run the application (production)
	@echo "🚀 Starting application in production mode..."
	docker-compose up -d
	@echo "✅ Application available at: http://localhost:8000"
	@echo "📊 File browser available at: http://localhost:8080 (run 'make tools' first)"

dev: ## Run in development mode with live reload
	@echo "🛠️  Starting application in development mode..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development server available at: http://localhost:8000"
	@echo "📁 File browser available at: http://localhost:8080"
	@echo "🔄 Live reload enabled for code changes"

prod: ## Run with production optimizations (nginx proxy)
	@echo "🏭 Starting application with production optimizations..."
	docker-compose --profile production up -d
	@echo "✅ Application available at: http://localhost (nginx proxy)"
	@echo "📊 Direct access: http://localhost:8000"

tools: ## Run with additional tools (file browser)
	@echo "🔧 Starting application with additional tools..."
	docker-compose --profile tools up -d
	@echo "✅ Application available at: http://localhost:8000"
	@echo "📁 File browser available at: http://localhost:8080"

# Management targets
stop: ## Stop all containers
	@echo "⏹️  Stopping containers..."
	docker-compose down
	docker-compose -f docker-compose.dev.yml down

clean: ## Stop containers and remove volumes
	@echo "🧹 Cleaning up containers and volumes..."
	docker-compose down -v
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f

reset: ## Full reset - remove everything
	@echo "🗑️  Performing full reset..."
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.dev.yml down -v --remove-orphans
	docker image rm prompt-library_prompt-library 2>/dev/null || true
	docker image rm prompt-library_prompt-library-dev 2>/dev/null || true
	docker system prune -af
	docker volume prune -f

# Utility targets
logs: ## View application logs
	docker-compose logs -f prompt-library

logs-dev: ## View development logs
	docker-compose -f docker-compose.dev.yml logs -f prompt-library-dev

shell: ## Open shell in running container
	docker-compose exec prompt-library bash

shell-dev: ## Open shell in development container
	docker-compose -f docker-compose.dev.yml exec prompt-library-dev bash

test: ## Run tests in container
	@echo "🧪 Running tests..."
	docker-compose exec prompt-library python /app/web_interface/test_enhanced_search.py

# Status targets
status: ## Show container status
	@echo "📊 Container Status:"
	docker-compose ps
	@echo ""
	@echo "🌐 Available Services:"
	@echo "  Main Application: http://localhost:8000"
	@echo "  API Documentation: http://localhost:8000/docs"
	@echo "  Health Check: http://localhost:8000/api/categories"

health: ## Check application health
	@echo "🏥 Checking application health..."
	@curl -s http://localhost:8000/api/categories >/dev/null && echo "✅ Application is healthy" || echo "❌ Application is not responding"

# Quick start targets
quick-start: build run status ## Build and run (quick start)
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo "   Open: http://localhost:8000"

dev-start: build-dev dev status ## Build and run in development mode
	@echo ""
	@echo "🎉 Development environment ready!"
	@echo "   Open: http://localhost:8000"
	@echo "   File browser: http://localhost:8080"

# Update targets
update: ## Pull latest changes and rebuild
	@echo "🔄 Updating application..."
	git pull
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Update complete!"