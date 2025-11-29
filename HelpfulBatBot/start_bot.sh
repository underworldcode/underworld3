#!/bin/bash
# HelpfulBatBot Startup Script

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🤖 HelpfulBatBot Startup"
echo "=================================================================="
echo ""

# Kill any existing bot instances on ports 8001-8010
echo "🧹 Cleaning up old instances..."
for port in {8001..8010}; do
    lsof -ti:$port | xargs kill -9 2>/dev/null
done
sleep 2

# Remove old port file
rm -f bot.port 2>/dev/null

# Start the bot
echo "🚀 Starting HelpfulBatBot..."
echo "   Model: Claude 3 Haiku"
echo "   Index: User-facing content only (86 files)"
echo ""

nohup python3 HelpfulBat_app.py > /tmp/helpfulbatbot.log 2>&1 &
BOT_PID=$!

echo "✅ Bot started (PID: $BOT_PID)"
echo "⏳ Waiting for bot to initialize and select port..."

# Wait for port file to be created (max 10 seconds)
MAX_WAIT=10
WAITED=0
while [ ! -f "bot.port" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
done

# Read the port
if [ -f "bot.port" ]; then
    BOT_PORT=$(cat bot.port)
    echo "✅ Bot selected port: $BOT_PORT"
    echo ""

    # Wait a bit more for the bot to be fully ready
    sleep 3

    # Check if it's responding
    if curl -s http://localhost:$BOT_PORT/health > /dev/null 2>&1; then
        echo "✅ Bot is ready!"
        echo ""
        echo "📝 Usage:"
        echo "   python3 ask.py \"Your question\""
        echo "   python3 ask.py status"
        echo ""
        echo "📊 Web interface:"
        echo "   http://localhost:$BOT_PORT/docs"
        echo ""
        echo "📋 Logs:"
        echo "   tail -f /tmp/helpfulbatbot.log"
    else
        echo "⚠️  Bot may still be starting. Check logs:"
        echo "   tail -f /tmp/helpfulbatbot.log"
    fi
else
    echo "⚠️  Port file not created. Check logs:"
    echo "   tail -f /tmp/helpfulbatbot.log"
fi
