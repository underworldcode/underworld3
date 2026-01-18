#!/usr/bin/env bash
# Build presentations for Underworld3
# Called standalone or from docs/build-docs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building UW3 presentations..."
quarto render

echo "Presentations built to: $SCRIPT_DIR/_build/"
