#!/bin/bash
# Quick local test script for HelpfulBatBot

set -e  # Exit on error

echo "🤖 HelpfulBatBot Local Test"
echo "======================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✅ Created .env - please edit it with your ANTHROPIC_API_KEY"
    echo ""
    echo "Get your key from: https://console.anthropic.com/settings/keys"
    echo "Then edit .env and run this script again"
    exit 1
fi

# Check if ANTHROPIC_API_KEY is set
source .env
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "sk-ant-xxxxxxxxxxxxx" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY not set in .env"
    echo "Please edit .env and add your Anthropic API key"
    exit 1
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import anthropic" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo "✅ Dependencies installed"
echo ""
echo "🚀 Starting HelpfulBatBot on http://localhost:8000"
echo "   Health check: http://localhost:8000/health"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 HelpfulBat_app.py
