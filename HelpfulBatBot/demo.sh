#!/bin/bash
# Quick demo of HelpfulBatBot

echo "═══════════════════════════════════════════════════════════════════"
echo "  🤖 HelpfulBatBot Demo - Underworld3 User Support Bot"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📍 Location: $(pwd)"
echo ""

# Start the bot
echo "🚀 Step 1: Starting the bot..."
./start_bot.sh
echo ""

# Wait for it to be ready
echo "⏳ Step 2: Waiting for bot to be ready (10 seconds)..."
sleep 10
echo ""

# Check status
echo "✅ Step 3: Checking bot status..."
python3 ask.py status
echo ""

# Ask a test question
echo "💬 Step 4: Asking a test question..."
echo "   Question: \"How do I create a mesh?\""
echo ""
python3 ask.py "How do I create a mesh in underworld3?"
echo ""

echo "═══════════════════════════════════════════════════════════════════"
echo "✨ Demo complete!"
echo ""
echo "📝 To ask your own questions:"
echo "   python3 ask.py \"Your question here\""
echo ""
echo "📖 For full documentation, read:"
echo "   cat README.md"
echo "═══════════════════════════════════════════════════════════════════"
