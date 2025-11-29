#!/usr/bin/env python3
"""Simple test that limits indexing to just a few files"""

import os
os.environ['BOT_MAX_FILE_SIZE'] = '50000'  # Smaller files only
os.environ['BOT_ALLOWED_EXTS'] = '.md'  # Only markdown files (faster)

import requests
import json

print("🤖 Simple HelpfulBatBot Test")
print("=" * 70)
print("⚙️  Config: Only indexing .md files under 50KB")
print("⏳ Sending test query...\n")

try:
    response = requests.post(
        "http://localhost:8001/ask",
        json={"question": "What is CLAUDE.md?", "max_context_items": 3},
        timeout=60  # 60 second timeout
    )

    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS!\n")
        print("=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(data['answer'][:500] + "..." if len(data['answer']) > 500 else data['answer'])
        print("\n" + "=" * 70)
        print("CITATIONS:")
        print("=" * 70)
        for citation in data['citations'][:3]:
            print(f"  - {citation}")
        print(f"\nUsed {len(data['used_files'])} files")
    else:
        print(f"❌ Error {response.status_code}")
        print(response.text[:200])

except requests.exceptions.Timeout:
    print("❌ Request timed out after 60 seconds")
    print("The bot is still indexing your large codebase.")
    print("\nTry restarting with a smaller file set:")
    print("  1. Stop the bot: pkill -f HelpfulBat_app")
    print("  2. Edit .env: BOT_ALLOWED_EXTS=.md")
    print("  3. Restart: python3 HelpfulBat_app.py &")

except Exception as e:
    print(f"❌ Error: {e}")
