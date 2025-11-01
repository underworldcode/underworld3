#!/bin/bash
# Live preview documentation build script for Underworld3
# Usage: ./watch-docs.sh
# This enables live reloading as you edit files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCS_DIR="$( dirname "$SCRIPT_DIR" )"

echo "=========================================="
echo "Underworld3 Documentation - Live Preview"
echo "=========================================="
echo ""
echo "📁 Docs Directory: $DOCS_DIR"
echo ""

# Check if quarto is available
if ! command -v quarto &> /dev/null; then
    echo "❌ Error: quarto not found in PATH"
    echo "Please ensure you're running with: pixi run -e default bash ./watch-docs.sh"
    exit 1
fi

echo "🔨 Starting live preview server..."
cd "$DOCS_DIR"

quarto preview --port 4173

echo ""
echo "✅ Live preview started"
echo ""
echo "📖 Open your browser to: http://localhost:4173"
echo "   Changes to .qmd and .ipynb files will auto-reload"
echo ""
echo "💡 Press Ctrl+C to stop the server"
