#!/usr/bin/env python3
"""
Tool to inspect what HelpfulBatBot has indexed
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Import bot's indexing code
import sys
sys.path.insert(0, '.')
from HelpfulBat_app import load_files, allowed_exts

print("🔍 HelpfulBatBot Index Inspector")
print("=" * 70)

repo_path = os.getenv('BOT_REPO_PATH')
print(f"📂 Repository: {repo_path}")
print(f"🎯 Allowed extensions: {', '.join(allowed_exts())}")
print(f"📏 Max file size: {os.getenv('BOT_MAX_FILE_SIZE', '200000')} bytes")
print()

files = load_files(repo_path)

print(f"📊 Found {len(files)} files")
print("=" * 70)

# Group by extension
from collections import Counter
exts = Counter(os.path.splitext(path)[1] for path, _ in files)

print("\n📁 Files by extension:")
for ext, count in exts.most_common():
    print(f"  {ext or '(no ext)'}: {count} files")

print("\n📝 Sample files:")
for i, (path, content) in enumerate(files[:10], 1):
    size_kb = len(content) / 1024
    print(f"  {i}. {path} ({size_kb:.1f} KB)")

if len(files) > 10:
    print(f"  ... and {len(files) - 10} more")

print("\n💡 To change what gets indexed:")
print("  1. Edit .env:")
print("     BOT_ALLOWED_EXTS=.md  # Only markdown")
print("     BOT_ALLOWED_EXTS=.py,.md  # Python and markdown")
print("     BOT_MAX_FILE_SIZE=50000  # Smaller files only")
print("  2. Restart the bot:")
print("     pkill -f HelpfulBat_app && python3 HelpfulBat_app.py &")
