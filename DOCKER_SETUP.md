# 🐳 Docker Setup Guide

Complete Docker environment for The Big Everything Prompt Library with all enhanced search capabilities.

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ available RAM
- 10GB+ available disk space

### One-Command Setup
```bash
# Clone and start (if not already cloned)
git clone <repository-url>
cd TheBigEverythingPromptLibrary

# Quick start with Make (recommended)
make quick-start

# OR manual start
docker-compose up -d
```

**Access Points:**
- 🌐 **Web Interface**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- 🔍 **Enhanced Search**: All features auto-enabled

## 🛠️ Available Commands

### Using Make (Recommended)
```bash
# Quick commands
make help           # Show all available commands
make quick-start    # Build and run production
make dev-start      # Build and run development mode
make stop           # Stop all containers
make clean          # Clean up containers and volumes

# Development
make dev            # Start development mode with live reload
make shell-dev      # Open shell in dev container
make logs-dev       # View development logs

# Production
make prod           # Start with nginx proxy
make tools          # Start with file browser
make test           # Run tests in container

# Maintenance
make update         # Pull changes and rebuild
make reset          # Full cleanup and reset
make status         # Show service status
make health         # Check application health
```

### Using Docker Compose Directly
```bash
# Production mode
docker-compose up -d                    # Start services
docker-compose logs -f prompt-library   # View logs
docker-compose down                     # Stop services

# Development mode
docker-compose -f docker-compose.dev.yml up -d    # Start dev
docker-compose -f docker-compose.dev.yml down     # Stop dev

# With additional services
docker-compose --profile tools up -d              # Add file browser
docker-compose --profile production up -d         # Add nginx proxy
```

## 🏗️ Environment Modes

### Production Mode (Default)
```bash
make run
# OR
docker-compose up -d
```

**Features:**
- ✅ Optimized Python image
- ✅ Pre-downloaded ML models
- ✅ Performance optimizations
- ✅ Health checks enabled
- ✅ Automatic restarts

**Access:**
- Main app: http://localhost:8000
- API docs: http://localhost:8000/docs

### Development Mode
```bash
make dev
# OR
docker-compose -f docker-compose.dev.yml up -d
```

**Features:**
- ✅ Live code reload
- ✅ Source code mounted
- ✅ Debug mode enabled
- ✅ Full file browser access
- ✅ Development tools

**Access:**
- Main app: http://localhost:8000 (with live reload)
- File browser: http://localhost:8080

### Production with Nginx
```bash
make prod
# OR
docker-compose --profile production up -d
```

**Features:**
- ✅ Nginx reverse proxy
- ✅ Rate limiting
- ✅ Gzip compression
- ✅ Security headers
- ✅ Load balancing ready

**Access:**
- Main app: http://localhost (nginx proxy)
- Direct access: http://localhost:8000

### With Additional Tools
```bash
make tools
# OR
docker-compose --profile tools up -d
```

**Features:**
- ✅ File browser interface
- ✅ Repository exploration
- ✅ File management (read-only in production)

**Access:**
- Main app: http://localhost:8000
- File browser: http://localhost:8080

## 📊 Service Overview

### Core Services

#### prompt-library
- **Purpose**: Main web application with enhanced search
- **Port**: 8000
- **Features**: All search capabilities, API endpoints, web UI
- **Health Check**: `/api/categories` endpoint

#### nginx (production profile)
- **Purpose**: Reverse proxy and performance optimization
- **Port**: 80
- **Features**: Rate limiting, compression, security headers

#### file-browser (tools profile)
- **Purpose**: Web-based file system browser
- **Port**: 8080
- **Features**: Repository exploration, file viewing

### Volume Management

#### Persistent Volumes
```bash
prompt-cache            # Embeddings and search cache
model-cache            # ML models and transformers cache
filebrowser-db         # File browser database
```

#### Mounted Volumes
```bash
.:/app:ro              # Repository content (read-only in production)
.:/app                 # Repository content (read-write in development)
```

## 🔧 Configuration

### Environment Variables
```bash
# Python configuration
PYTHONPATH=/app/.scripts:/app/web_interface/backend
PYTHONUNBUFFERED=1

# ML model cache
TORCH_HOME=/app/.cache/torch
HF_HOME=/app/.cache/huggingface
TRANSFORMERS_CACHE=/app/.cache/huggingface

# Development mode
DEBUG=true                    # Enable debug mode
PYTHONDONTWRITEBYTECODE=1     # Disable .pyc files
```

### Resource Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Disk**: 10GB

#### Recommended for Full Features
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Disk**: 20GB+

### Performance Tuning

#### For Better Search Performance
```bash
# Increase shared memory for neural models
docker-compose up -d --shm-size=2g
```

#### For Large Repositories
```bash
# Increase container memory
docker-compose up -d --memory=8g
```

## 🧪 Testing and Debugging

### Run Tests
```bash
# In running container
make test

# Standalone test
docker-compose exec prompt-library python /app/web_interface/test_enhanced_search.py
```

### Debug Mode
```bash
# Start development mode
make dev

# Open shell for debugging
make shell-dev

# View detailed logs
make logs-dev
```

### Health Checks
```bash
# Check service health
make health

# Manual health check
curl http://localhost:8000/api/categories

# Check all services
make status
```

## 🐛 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
make logs

# Check system resources
docker system df
docker stats

# Clean and rebuild
make reset
make build
```

#### Search Features Not Working
```bash
# Check if models downloaded
make shell
ls -la /app/.cache/huggingface/

# Test enhanced search
make test

# Check dependencies
docker-compose exec prompt-library pip list | grep sentence
```

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000

# Use different ports
export PORT=8080
docker-compose up -d
```

#### Out of Memory
```bash
# Check memory usage
docker stats

# Reduce memory usage
docker-compose up -d --scale prompt-library=1 --memory=4g
```

### Reset Everything
```bash
# Complete cleanup
make reset

# Rebuild from scratch
make build
make run
```

## 🚀 Deployment Options

### Local Development
```bash
make dev-start
```

### Local Production
```bash
make prod
```

### Server Deployment
```bash
# Clone repository
git clone <repository-url>
cd TheBigEverythingPromptLibrary

# Set production environment
export COMPOSE_FILE=docker-compose.yml
export COMPOSE_PROFILES=production

# Start services
docker-compose up -d

# Configure reverse proxy (nginx/traefik)
# Point to http://server-ip:8000
```

### Cloud Deployment
```bash
# Use provided docker-compose.yml
# Configure cloud volumes for persistence
# Set up load balancer to port 8000
# Configure SSL termination
```

## 📈 Monitoring

### Check Service Status
```bash
make status
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
make logs
```

### Monitor Resources
```bash
# Container stats
docker stats

# System usage
docker system df
```

## 🔄 Updates and Maintenance

### Update Application
```bash
make update
```

### Manual Update
```bash
git pull
docker-compose build --no-cache
docker-compose up -d
```

### Backup Data
```bash
# Backup volumes
docker run --rm -v prompt-cache:/source -v $(pwd):/backup alpine tar czf /backup/prompt-cache.tar.gz -C /source .

# Backup configurations
cp docker-compose.yml docker-compose.yml.backup
```

### Restore Data
```bash
# Restore volumes
docker run --rm -v prompt-cache:/target -v $(pwd):/backup alpine tar xzf /backup/prompt-cache.tar.gz -C /target
```

---

## 🎉 You're Ready!

Your Docker environment includes:
- ✅ **Enhanced Search Engine** with quality scoring
- ✅ **Semantic Search** capabilities
- ✅ **Fuzzy Matching** for typo tolerance
- ✅ **AI Enhancement Tools** (optional)
- ✅ **Production-ready Setup** with health checks
- ✅ **Development Environment** with live reload
- ✅ **File Browser** for repository exploration
- ✅ **Nginx Proxy** for production deployment

Start with `make quick-start` and explore the enhanced search capabilities at http://localhost:8000!