# Claude Integration for HelpfulBatBot

## What Changed

Your HelpfulBatBot has been updated to use **Claude 3.5 Sonnet with prompt caching** instead of generic OpenAI-style endpoints.

### Key Improvements

1. **Better Code Understanding**
   - Claude 3.5 Sonnet excels at reading complex codebases
   - Understands PETSc, parallel computing, finite element methods
   - More accurate citations and code examples

2. **90% Cost Reduction with Prompt Caching**
   - First query: Standard cost ($3/million input tokens)
   - Subsequent queries (within 5 minutes): 90% cheaper ($0.30/million tokens)
   - Perfect for a bot answering many questions about the same codebase

3. **UW3-Specific System Prompt**
   - Knows about parallel safety (`uw.pprint()`, `uw.selective_ranks()`)
   - References `CLAUDE.md` for architectural guidelines
   - Mentions rebuild requirements (`pixi run underworld-build`)
   - Warns about solver stability

4. **Health Check Endpoint**
   - `GET /health` shows bot status, model info, document count
   - Useful for monitoring and Fly.io deployments

### Modified Files

- **HelpfulBat_app.py**:
  - Added `import anthropic`
  - Replaced `call_llm()` with `call_llm_with_caching()`
  - Enhanced system prompt for UW3 expertise
  - Added `/health` endpoint

- **.env.example**:
  - Changed to use `ANTHROPIC_API_KEY` instead of generic `BOT_LLM_ENDPOINT`/`BOT_LLM_API_KEY`
  - Added `CLAUDE_MODEL` option (defaults to `claude-3-5-sonnet-20241022`)

### New Files

- **test_locally.sh**: One-command local testing
- **test_query.sh**: Send test queries to running bot
- **CLAUDE_INTEGRATION.md**: This file

---

## How to Test Locally

### Step 1: Set up your API key

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot

# Create .env from template
cp .env.example .env

# Edit .env and add your Anthropic API key
# Get one from: https://console.anthropic.com/settings/keys
nano .env  # or vim, code, etc.
```

Make sure to set:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxx
BOT_REPO_PATH=/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
```

### Step 2: Run the bot

```bash
./test_locally.sh
```

This will:
- Check if `.env` exists and is configured
- Install dependencies (`pip install -r requirements.txt`)
- Start the FastAPI server on `http://localhost:8000`

You should see:
```
🤖 HelpfulBatBot Local Test
======================
📦 Checking dependencies...
✅ Dependencies installed

🚀 Starting HelpfulBatBot on http://localhost:8000
   Health check: http://localhost:8000/health
   Docs: http://localhost:8000/docs
```

### Step 3: Test queries (in another terminal)

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/HelpfulBatBot

# Test with default question
./test_query.sh

# Or ask custom questions
./test_query.sh "How do I rebuild underworld3 after changing source files?"
./test_query.sh "What is the parallel safety system?"
./test_query.sh "How do I create a Stokes solver?"
```

You should see JSON output with:
- `answer`: Claude's response with citations
- `citations`: Links to GitHub files
- `used_files`: Files that were referenced
- `confidence`: 0.5-0.8 confidence score

---

## Expected Behavior

### First Query (Building Index)
The first query will take ~10-30 seconds because:
1. SentenceTransformer loads the embedding model (~100MB)
2. FAISS indexes all UW3 files
3. Claude processes the context

Subsequent queries are much faster (~2-5 seconds).

### Successful Response Example

```json
{
  "answer": "To use parallel-safe printing in Underworld3, use `uw.pprint()`:\n\n```python\nimport underworld3 as uw\n\n# Only rank 0 prints, but all ranks evaluate the expression\nuw.pprint(0, f\"Mesh has {mesh.data.shape[0]} local nodes\")\n\n# Multiple ranks can print\nuw.pprint([0,1,2], \"First three ranks reporting\")\n```\n\nSee CLAUDE.md:109-131 for complete documentation.",
  "citations": [
    "https://github.com/underworldcode/underworld3/blob/main/CLAUDE.md#L109-L131",
    "https://github.com/underworldcode/underworld3/blob/main/src/underworld3/mpi.py#L45-L78"
  ],
  "used_files": [
    "CLAUDE.md",
    "src/underworld3/mpi.py"
  ],
  "confidence": 0.8
}
```

### Health Check Response

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "index_built": true,
  "doc_count": 427,
  "embedding_model": "all-MiniLM-L6-v2",
  "claude_model": "claude-3-5-sonnet-20241022"
}
```

---

## Troubleshooting

### "I don't have Claude configured"

**Problem**: Bot can't find `ANTHROPIC_API_KEY`

**Solution**:
```bash
# Check if .env exists
ls -la .env

# Check if key is set
cat .env | grep ANTHROPIC_API_KEY

# If not set, edit .env and add your key
```

### "ModuleNotFoundError: No module named 'anthropic'"

**Problem**: Dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt

# Or install just anthropic
pip install anthropic
```

### "BOT_REPO_PATH not set"

**Problem**: Bot can't find UW3 repository

**Solution**: Edit `.env` and set:
```bash
BOT_REPO_PATH=/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3
```

### Index takes forever / runs out of memory

**Problem**: Indexing too many large files

**Solution**: Increase `BOT_MAX_FILE_SIZE` limit or exclude large files:
```bash
# In .env
BOT_MAX_FILE_SIZE=100000  # Smaller limit (100KB)
```

### "No documents indexed"

**Problem**: No matching files found

**Solution**: Check that `BOT_REPO_PATH` points to the correct directory:
```bash
ls $BOT_REPO_PATH/src/underworld3/  # Should see Python files
```

---

## Cost Estimate

Assuming 1000 queries/month with similar context:

| Component | Cost |
|-----------|------|
| **First query** | $0.015 (5K context @ $3/M tokens) |
| **Cached queries (999)** | $0.015 (999 × $0.000015) |
| **Output tokens** | ~$2 (assuming 500 tokens/response @ $15/M) |
| **Total/month** | **~$2-3** |

Compare to GPT-4o-mini (no caching): ~$5-10/month

---

## Model Comparison

| Model | Model ID | Use Case | Input Cost | Caching |
|-------|----------|----------|------------|---------|
| **Claude 3.5 Sonnet** ✅ | `claude-3-5-sonnet-20241022` | **Code Q&A (recommended)** | $3/M → $0.30/M | Yes |
| Claude 3.5 Haiku | `claude-3-5-haiku-20241022` | Fast, simple queries | $0.80/M → $0.08/M | Yes |
| Claude 3 Opus | `claude-3-opus-20240229` | Maximum quality | $15/M → $1.50/M | Yes |

To switch models, edit `.env`:
```bash
CLAUDE_MODEL=claude-3-5-haiku-20241022  # Faster, cheaper
# or
CLAUDE_MODEL=claude-3-opus-20240229  # Highest quality
```

---

## Next Steps

1. ✅ **Test locally** (you're here!)
2. 🚀 **Deploy to Fly.io** (see `DEPLOYMENT.md`)
3. 🔗 **Update GitHub workflow** (point to your deployed URL)
4. 💬 **Add to documentation** (chat widget for UW3 docs)
5. 📊 **Monitor usage** (check Anthropic console for API usage)

---

## Reverting to Generic LLM

If you need to use OpenAI or another LLM, you can revert:

1. Replace `call_llm_with_caching()` with the old generic `call_llm()`
2. Update `.env` to use `BOT_LLM_ENDPOINT` and `BOT_LLM_API_KEY`
3. Change `import anthropic` to `import requests`

Or keep both versions and switch via environment variable!

---

## Support

- **Anthropic API Docs**: https://docs.anthropic.com/
- **Prompt Caching**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- **Claude Models**: https://docs.anthropic.com/en/docs/about-claude/models

For HelpfulBatBot issues, see `HelpfulBat_README.md` and `DEPLOYMENT.md`.
