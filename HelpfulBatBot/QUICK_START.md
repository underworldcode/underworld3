# HelpfulBatBot Quick Start

**Location:** `/Users/lmoresi/+Underworld/underworld3-helpfulbat-bot`

## Run the Demo (Easiest!)

```bash
./demo.sh
```

This will start the bot and ask a test question to show you how it works.

## Manual Usage

### 1. Start the Bot
```bash
./start_bot.sh
```

### 2. Ask Questions
```bash
python3 ask.py "How do I create a mesh?"
python3 ask.py "What is uw.pprint?"
python3 ask.py status
```

## All Files in This Directory

**Main Tools:**
- `ask.py` - Ask the bot questions (USE THIS!)
- `start_bot.sh` - Start the bot
- `demo.sh` - Full demo of the bot
- `HelpfulBat_app.py` - The bot server

**Documentation:**
- `README.md` - Complete guide
- `QUICK_START.md` - This file
- `IMPLEMENTATION_SUMMARY.md` - Technical details

**Testing:**
- `inspect_index.py` - See what files are indexed
- `test_new_index.py` - Test path filtering

**Configuration:**
- `.env` - Bot settings (API key, paths, etc.)

## Example Questions to Try

```bash
python3 ask.py "How do I create a mesh?"
python3 ask.py "What is the units system in UW3?"
python3 ask.py "How do I use parallel computing?"
python3 ask.py "How do I set up a Stokes solver?"
python3 ask.py "What are swarms in UW3?"
```

## Troubleshooting

**Bot not responding?**
```bash
./start_bot.sh
```

**Want to see what's indexed?**
```bash
python3 inspect_index.py
```

**Need help?**
```bash
cat README.md
```

## What the Bot Knows About

✅ Tutorial notebooks (15 files)
✅ Example scripts (23 files)
✅ A/B grade tests (35 files)
✅ User documentation (24 files)

❌ Source code internals (excluded)
❌ Developer documentation (excluded)

---

**Next Step:** Run `./demo.sh` to see it in action!
