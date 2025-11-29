# Auto-Port Detection Feature

## Overview

HelpfulBatBot now automatically detects and uses an available port, making it more robust when port 8001 is already in use.

## How It Works

### 1. Bot Startup (`HelpfulBat_app.py`)

When the bot starts, it:
1. Searches for an available port starting from 8001 (tries ports 8001-8010)
2. Writes the selected port to `bot.port` file
3. Starts the server on the detected port

```python
# Port detection function
def find_available_port(start_port=8001, max_attempts=10):
    """Try ports from 8001-8010 until finding one that's free"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found")
```

### 2. Client Connection (`ask.py`)

When you run `ask.py`, it:
1. Reads the port number from `bot.port` file
2. Connects to `http://localhost:{port}`
3. Falls back to port 8001 if `bot.port` doesn't exist

```python
def get_bot_port(default_port=8001):
    """Read port from bot.port file, or use default"""
    port_file = Path(__file__).parent / "bot.port"
    if port_file.exists():
        return int(port_file.read_text().strip())
    return default_port
```

### 3. Startup Script (`start_bot.sh`)

The startup script:
1. Kills any existing bot processes on ports 8001-8010
2. Removes old `bot.port` file
3. Starts the bot and waits for `bot.port` to be created
4. Reads the port and verifies the bot is responding
5. Shows the dynamic web interface URL

## Usage

### Normal Usage (No Changes Needed!)

```bash
# Start the bot
./start_bot.sh

# Ask questions
python3 ask.py "How do I create a mesh?"
python3 ask.py status
```

### What You'll See

```bash
$ ./start_bot.sh
🤖 HelpfulBatBot Startup
==================================================================

🧹 Cleaning up old instances...
🚀 Starting HelpfulBatBot...
   Model: Claude 3 Haiku
   Index: User-facing content only (86 files)

✅ Bot started (PID: 12345)
⏳ Waiting for bot to initialize and select port...
✅ Bot selected port: 8001
✅ Bot is ready!

📝 Usage:
   python3 ask.py "Your question"
   python3 ask.py status

📊 Web interface:
   http://localhost:8001/docs
```

If port 8001 is busy:
```bash
✅ Bot selected port: 8002  # Automatically uses next available port
```

## Benefits

✅ **No manual port configuration** - Works automatically
✅ **Handles port conflicts** - Tries up to 10 ports (8001-8010)
✅ **Transparent to users** - `ask.py` automatically finds the bot
✅ **Multiple instances** - Can run multiple bots simultaneously (different directories)
✅ **Backward compatible** - Still defaults to 8001 when possible

## Files Modified

1. **`HelpfulBat_app.py`**:
   - Added `find_available_port()` function
   - Added `write_port_file()` function
   - Updated main block to use auto-detection

2. **`ask.py`**:
   - Added `get_bot_port()` function
   - Updated `ask_bot()` to use dynamic port
   - Updated `show_status()` to use dynamic port

3. **`start_bot.sh`**:
   - Kills processes on ports 8001-8010
   - Waits for `bot.port` file to be created
   - Reads and displays the selected port
   - Uses dynamic port for health check

4. **`.gitignore`**:
   - Added `bot.port` to prevent committing runtime file
   - Added `bot.pid` for consistency

## Port File Format

The `bot.port` file contains a single line with the port number:
```
8001
```

This file is:
- Created automatically when the bot starts
- Read automatically by `ask.py`
- Ignored by git (listed in `.gitignore`)
- Removed by `start_bot.sh` before starting a new instance

## Troubleshooting

### Bot can't find an available port
**Error**: `RuntimeError: No available ports found in range 8001-8010`

**Solution**: All 10 ports are in use. Either:
- Stop some services: `lsof -ti:8001 | xargs kill -9`
- Increase `max_attempts` in `HelpfulBat_app.py`

### ask.py can't find the bot
**Symptom**: `ask.py` shows connection error

**Check**:
```bash
# Is bot.port file present?
cat bot.port

# Is the bot actually running?
lsof -i :8001  # or whatever port is in bot.port
```

### Multiple bots running
If you accidentally start multiple bots:
```bash
# Kill all instances
for port in {8001..8010}; do lsof -ti:$port | xargs kill -9 2>/dev/null; done

# Start fresh
./start_bot.sh
```

## Technical Details

### Why ports 8001-8010?

- **8001**: Default, likely to be free
- **8002-8010**: Fallback range for conflicts
- **10 attempts**: Enough for typical use cases without being excessive

### Socket Testing Method

The `find_available_port()` function uses Python's `socket.bind()` to test port availability:
- **Advantage**: Fast, reliable, cross-platform
- **Limitation**: Port could be taken between test and actual server start (rare race condition)
- **Mitigation**: Uvicorn will error immediately if port is taken, easy to debug

### Port File vs Environment Variables

We chose a port file over environment variables because:
- ✅ Works across different shells
- ✅ Survives shell restarts
- ✅ Easy to inspect (`cat bot.port`)
- ✅ Simple to clean up (`rm bot.port`)

---

**Date**: November 18, 2025
**Feature**: Auto-port detection for robust bot deployment
