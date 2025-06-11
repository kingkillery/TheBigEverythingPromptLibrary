# 🌐 Web Interface & API

FastAPI-powered backend and lightweight frontend for exploring **The Big Everything Prompt Library** with advanced search and LLM-assisted tools.

| Endpoint | Description |
|----------|-------------|
| `/` | SPA frontend (static HTML + HTMX/Alpine.js) |
| `/docs` | Interactive Swagger / OpenAPI UI |
| `/api/search` | Hybrid / semantic search with filters |
| `/api/search/details` | Per-item score breakdown + stats |
| `/api/item/{id}` | Full markdown for a prompt |
| `/api/similar/{id}` | Semantically similar prompts |
| `/api/collections` | Personal "garden beds" of favourite prompts |
| `/api/llm/*` | Prompt improvement / summarisation (requires OpenRouter key) |
| `/api/trending-feed` | Trending articles from popular networks |

---

## 🚀 Quick Start (Docker)

```powershell
# clone & run
git clone https://github.com/kingkillery/TheBigEverythingPromptLibrary.git
cd TheBigEverythingPromptLibrary
docker compose up -d        # or: make quick-start
```

* Open **http://localhost:8000** for the UI  
* Open **http://localhost:8000/docs** for the API docs (Swagger UI)

For advanced profiles (nginx proxy, file-browser, dev live-reload) see the [Docker Setup Guide](../DOCKER_SETUP.md).

---

## 🧑‍💻 Local Dev Without Docker

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r web_interface/requirements.txt
python -m uvicorn web_interface.backend.app:app --reload
```

Navigate to http://localhost:8000 ‑ hot-reload is enabled for changes inside `web_interface/`.

---

## ✨ Features at a Glance

- **Hybrid search** (keyword + semantic + quality scoring)
- **Suggestion engine** (`/api/search/suggestions`)
- **Quality distribution analytics** (`/api/quality-filter`)
- **Collections API** for saving prompt sets (`X-User-Id` header)
- **LLM utilities** (improve, tag, summarise, compare) – opt-in after adding an OpenRouter key

---

## 🔑 Supplying an OpenRouter API Key (optional)

```bash
POST /api/llm/set-key
{
  "api_key": "sk-or-..."
}
```

Afterwards list available models at `/api/llm/models`.

---

## 📈 Health Check

```powershell
Invoke-WebRequest http://localhost:8000/api/ping | ConvertFrom-Json
```

Returns JSON with status, version and number of indexed items.

---

Made with ❤️ & FastAPI.