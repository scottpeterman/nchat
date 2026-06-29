#!/bin/bash
# nChat - Quick start script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════╗${NC}"
echo -e "${BLUE}║       nChat          ║${NC}"
echo -e "${BLUE}╚══════════════════════╝${NC}"
echo ""

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama not found. Install: curl -fsSL https://ollama.ai/install.sh | sh${NC}"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Ollama not running. Starting...${NC}"
    ollama serve &
    sleep 2
fi

echo -e "${GREEN}✓ Ollama connected${NC}"

# Python venv
if [ ! -d "venv" ]; then
    echo -e "${BLUE}→ Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# Check frontend build
if [ ! -d "frontend/dist" ]; then
    echo -e "${YELLOW}⚠ Frontend not built. Building...${NC}"
    cd frontend
    npm install
    npm run build
    cd ..
fi

echo -e "${GREEN}✓ Starting nChat on http://localhost:8400${NC}"
echo ""

uvicorn backend.main:app --host 0.0.0.0 --port 8400
