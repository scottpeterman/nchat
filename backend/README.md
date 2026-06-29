# nChat

Local LLM chat interface powered by Ollama. FastAPI backend + React frontend.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend build)
- **Ollama** running locally with at least one model pulled

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a coding model
ollama pull qwen2.5-coder:32b
```

## Quick Start

### 1. Backend

```bash
cd nchat
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the API server
uvicorn backend.main:app --host 0.0.0.0 --port 8400 --reload
```

### 2. Frontend (Development)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** — the Vite dev server proxies API calls to the FastAPI backend.

### Production Build

```bash
cd frontend
npm run build
```

Then just run the FastAPI server — it serves the built frontend from `frontend/dist/`:

```
uvicorn backend.main:app --host 0.0.0.0 --port 8400
```

Open **http://localhost:8400**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + Ollama status |
| GET | `/api/models` | List available Ollama models |
| POST | `/api/chat` | Stream chat (SSE) |
| GET | `/api/conversations` | List conversations |
| POST | `/api/conversations` | Create conversation |
| GET | `/api/conversations/{id}` | Get conversation |
| PUT | `/api/conversations/{id}` | Update conversation |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| GET | `/api/conversations/{id}/messages` | Get messages |

## Architecture

```
nchat/
├── backend/
│   ├── main.py          # FastAPI app, Ollama proxy, SSE streaming
│   ├── database.py      # SQLite persistence layer
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app state management
│   │   ├── components/
│   │   │   ├── ChatView.jsx     # Message display + input
│   │   │   ├── MessageBubble.jsx # Markdown + syntax highlighting
│   │   │   ├── ModelSelector.jsx # Model dropdown
│   │   │   └── Sidebar.jsx      # Conversation history
│   │   └── styles/
│   │       └── app.css          # Dark theme
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
└── README.md
```

## Features

- **Streaming responses** via Server-Sent Events
- **Conversation history** persisted in SQLite
- **Model selection** from available Ollama models
- **Markdown rendering** with syntax-highlighted code blocks
- **Copy code** button on all code blocks
- **Performance stats** (tokens/sec, duration) per response
- **Auto-titling** conversations from first message
- **Dark theme** with JetBrains Mono + IBM Plex Sans

## License

MIT
