#!/usr/bin/env python3
"""Quick test of HelpfulBatBot"""

import requests
import json

question = "How do I use uw.pprint for parallel-safe printing?"

print(f"🤖 Testing HelpfulBatBot...")
print(f"📝 Question: {question}")
print(f"⏳ Sending request (this may take 10-30 seconds for first query)...\n")

response = requests.post(
    "http://localhost:8001/ask",
    json={"question": question, "max_context_items": 6}
)

if response.status_code == 200:
    data = response.json()
    print("✅ Success!\n")
    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(data['answer'])
    print("\n" + "=" * 70)
    print("CITATIONS:")
    print("=" * 70)
    for citation in data['citations']:
        print(f"  - {citation}")
    print("\n" + "=" * 70)
    print(f"Used files: {', '.join(data['used_files'])}")
    print(f"Confidence: {data['confidence']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
