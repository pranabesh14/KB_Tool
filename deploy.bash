#!/usr/bin/env bash
# =============================================================================
# deploy.sh — KB Tool setup and launch script for Linux / macOS
# Usage:
#   chmod +x deploy.sh   (first time only)
#   ./deploy.sh
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "   KB Tool — Setup & Launch"
echo "============================================================"
echo ""

# ── 0. Resolve script directory so the script works from any CWD ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Check Python 3.9+ is available ────────────────────────────────────────
info "[1/5] Checking Python installation..."

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_BIN="$candidate"
            ok "Found: $candidate $version"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "Python 3.9+ not found. Install it from https://www.python.org/downloads/"
fi
echo ""

# ── 2. Create virtual environment ─────────────────────────────────────────────
info "[2/5] Setting up virtual environment..."

if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
    ok "Virtual environment created at .venv/"
else
    ok ".venv/ already exists, skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate
echo ""

# ── 3. Install requirements ───────────────────────────────────────────────────
info "[3/5] Installing requirements..."

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

ok "All packages installed successfully."
echo ""

# ── 4. Create required directories ───────────────────────────────────────────
info "[4/5] Creating required directories..."

DIRS=("data" "faiss_pdf" "faiss_video" "qa_indexes" "modules")

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        ok "Created: $dir/"
    else
        info "Already exists: $dir/"
    fi
done
echo ""

# ── 5. Check .env file ────────────────────────────────────────────────────────
info "[5/5] Checking environment configuration..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        warn ".env not found — copied from .env.example."
        warn "Please edit .env and fill in your Azure OpenAI credentials before using the app."
    else
        warn "Neither .env nor .env.example found."
        warn "Create a .env file with your Azure OpenAI credentials."
    fi
else
    ok ".env found."
fi
echo ""

# ── 6. Launch Streamlit app ───────────────────────────────────────────────────
echo "============================================================"
echo "   Launching KB Tool..."
echo "   Open your browser at: http://localhost:8501"
echo "   Press Ctrl+C to stop the app."
echo "============================================================"
echo ""

streamlit run app.py --server.port 8501 --server.headless false