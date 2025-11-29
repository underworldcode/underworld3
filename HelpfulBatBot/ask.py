#!/usr/bin/env python3
"""
Friendly CLI tool to interact with HelpfulBatBot
Usage: python3 ask.py "Your question here"
"""

import sys
import requests
import json
from pathlib import Path

def get_bot_port(default_port=8001):
    """
    Read the bot port from bot.port file.

    Args:
        default_port: Port to use if file doesn't exist (default: 8001)

    Returns:
        int: Port number to connect to
    """
    port_file = Path(__file__).parent / "bot.port"
    if port_file.exists():
        try:
            port = int(port_file.read_text().strip())
            return port
        except (ValueError, OSError):
            print(f"⚠️  Warning: Could not read port from {port_file}, using default {default_port}")
            return default_port
    else:
        print(f"ℹ️  Port file not found, using default port {default_port}")
        return default_port

def ask_bot(question, num_context=6):
    """Ask HelpfulBatBot a question"""

    print(f"🤖 HelpfulBatBot")
    print("=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)
    print("⏳ Thinking...\n")

    # Get bot port
    port = get_bot_port()

    try:
        response = requests.post(
            f"http://localhost:{port}/ask",
            json={"question": question, "max_context_items": num_context},
            timeout=180  # 3 minutes for first query (builds index)
        )

        if response.status_code == 200:
            data = response.json()

            print("📝 ANSWER:")
            print("-" * 70)
            print(data['answer'])
            print()

            if data.get('citations'):
                print("📚 CITATIONS:")
                print("-" * 70)
                for i, citation in enumerate(data['citations'], 1):
                    print(f"{i}. {citation}")
                print()

            if data.get('used_files'):
                print("📂 FILES USED:")
                print("-" * 70)
                for f in data['used_files']:
                    print(f"  • {f}")
                print()

            print(f"✨ Confidence: {data.get('confidence', 'unknown')}")

        else:
            print(f"❌ Error {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print("❌ Request timed out. The bot might still be indexing.")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to bot. Is it running?")
        print("   Start it with: python3 HelpfulBat_app.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def show_status():
    """Show bot status and indexed files"""
    port = get_bot_port()
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("🤖 HelpfulBatBot Status")
            print("=" * 70)
            print(f"Status: {data['status']}")
            print(f"Index built: {data['index_built']}")
            print(f"Documents indexed: {data['doc_count']}")
            print(f"Embedding model: {data['embedding_model']}")
            print(f"Claude model: {data['claude_model']}")
        else:
            print("❌ Bot returned error")
    except:
        print("❌ Bot not responding. Is it running?")
        print("   Start it with: python3 HelpfulBat_app.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("HelpfulBatBot - Interactive UW3 Assistant")
        print("=" * 70)
        print()
        print("Usage:")
        print("  python3 ask.py \"Your question\"")
        print("  python3 ask.py status")
        print()
        print("Examples:")
        print('  python3 ask.py "How do I use uw.pprint?"')
        print('  python3 ask.py "What is CLAUDE.md?"')
        print('  python3 ask.py "How do I rebuild underworld3?"')
        print('  python3 ask.py status')
        print()
        sys.exit(0)

    if sys.argv[1].lower() == "status":
        show_status()
    else:
        question = " ".join(sys.argv[1:])
        ask_bot(question)
