# HelpfulBatBot - Underworld3 User Support Bot

Location: `/Users/lmoresi/+Underworld/underworld3-helpfulbat-bot`

## Quick Start (3 Steps)

### 1. Start the Bot
```bash
./start_bot.sh
```
Wait 2-3 minutes for indexing on first use.

### 2. Ask Questions
```bash
python3 ask.py "How do I create a mesh?"
python3 ask.py "What is uw.pprint?"
python3 ask.py "How do I use parallel computing?"
```

### 3. Check Status
```bash
python3 ask.py status
```

## All Available Scripts

### Main Usage
- **`ask.py`** - Ask the bot questions (easiest way to use it)
- **`start_bot.sh`** - Start the bot server
- **`HelpfulBat_app.py`** - The bot server itself (runs on port 8001)

### Testing & Inspection
- **`inspect_index.py`** - See what files are indexed
- **`test_new_index.py`** - Test the path filtering logic
- **`analyze_content.py`** - Analyze UW3 content structure

### Documentation
- **`IMPLEMENTATION_SUMMARY.md`** - Complete technical overview
- **`DEPLOYMENT.md`** - How to deploy to production
- **`CLAUDE_INTEGRATION.md`** - Technical details on Claude integration

## Example Session

```bash
# Start the bot
./start_bot.sh

# Wait 2-3 minutes, then ask questions
python3 ask.py "How do I create a mesh in underworld3?"

# Output:
# 🤖 HelpfulBatBot
# ❓ Question: How do I create a mesh in underworld3?
#
# 📝 ANSWER:
# To create a mesh in Underworld3, use the uw.meshing module...
# [Complete answer with code examples and citations]
```

## What's Indexed

The bot indexes **ONLY user-facing content** (86 files):
- ✅ 15 tutorial notebooks
- ✅ 23 example scripts
- ✅ 35 A/B grade tests
- ✅ 24 documentation files

**Excluded** (not indexed):
- ❌ Source code internals (src/)
- ❌ Developer docs (docs/developer/)
- ❌ Planning docs (planning/)
- ❌ Build artifacts

## Configuration

All settings in `.env`:
```bash
BOT_REPO_PATH=/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
ANTHROPIC_API_KEY=sk-ant-api03-...
CLAUDE_MODEL=claude-3-haiku-20240307
```

## Troubleshooting

**Bot not responding?**
```bash
# Check if it's running
curl http://localhost:8001/health

# Restart it
./start_bot.sh
```

**First query taking forever?**
- Normal! Index builds on first query (~2 minutes for 86 files)
- Subsequent queries are fast (5-10 seconds)

**Want to change what's indexed?**
- Edit patterns in `.env` under `BOT_INCLUDE_PATHS` and `BOT_EXCLUDE_PATHS`
- Restart the bot: `./start_bot.sh`

## Web Interface

While the bot is running, visit:
- **API docs:** http://localhost:8001/docs
- **Health check:** http://localhost:8001/health

## Stopping the Bot

```bash
lsof -ti:8001 | xargs kill -9
```

## Cost

With Claude prompt caching: **~1 cent per answer**
Monthly cost for 1000 queries: **~$2-3**

## Next Steps

1. Test it with real UW3 questions
2. Deploy to Fly.io for GitHub integration (see DEPLOYMENT.md)
3. Add rate limiting for production use
