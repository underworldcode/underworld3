#!/bin/bash
# Test HelpfulBatBot with a sample query

QUESTION="${1:-How do I use uw.pprint for parallel-safe printing?}"

echo "❓ Testing HelpfulBatBot with question:"
echo "   \"$QUESTION\""
echo ""

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"$QUESTION\", \"max_context_items\": 6}" \
  2>/dev/null | python3 -m json.tool

echo ""
echo "✅ Done! Try other questions:"
echo "   ./test_query.sh \"How do I rebuild underworld3?\""
echo "   ./test_query.sh \"What is the parallel safety system?\""
echo "   ./test_query.sh \"How do I create a Stokes solver?\""
