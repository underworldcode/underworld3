#!/usr/bin/env bash

# Build script for Underworld3 documentation
# Builds presentations first (so they can be embedded), then the book

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building Underworld3 Documentation ==="

# Build presentations first
echo ""
echo "--- Building presentations ---"
./presentations/build.sh

# Build the book
echo ""
echo "--- Building documentation book ---"
quarto render

echo ""
echo "=== Build complete ==="
echo "Output:"
echo "  - Book:   _build/index.html"
echo "  - Slides: presentations/_build/"
