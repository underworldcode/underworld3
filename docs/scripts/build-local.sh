#!/bin/bash
# Local documentation build script for Underworld3
# Usage: ./build-local.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCS_DIR="$( dirname "$SCRIPT_DIR" )"

echo "=========================================="
echo "Underworld3 Documentation Build"
echo "=========================================="
echo ""
echo "📁 Docs Directory: $DOCS_DIR"
echo ""

# Check if quarto is available
if ! command -v quarto &> /dev/null; then
    echo "❌ Error: quarto not found in PATH"
    echo "Please ensure you're running with: pixi run -e default bash ./build-local.sh"
    exit 1
fi

echo "🔨 Building documentation..."
cd "$DOCS_DIR"

# Run quarto render
quarto render . --to html

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📖 Documentation built to: _build/"
    echo ""
    echo "💡 To view locally, open: _build/index.html"
    echo "   Or use: pixi run docs-watch for live preview"
    echo ""
else
    echo ""
    echo "❌ Build failed. Check output above for errors."
    exit 1
fi
