# HelpfulBatBot Deployment Guide

This directory is **self-contained** - everything needed to deploy the Underworld3 support bot is here.

## Quick Start (Local Testing)

1. **Set up environment**:
   ```bash
   cd HelpfulBatBot
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**:
   ```bash
   python HelpfulBat_app.py
   # Bot runs at http://localhost:8000
   ```

4. **Test the bot**:
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I use uw.pprint?", "max_context_items": 6}'
   ```

---

## Deploy to Fly.io (Zero-Config)

### Prerequisites

1. **Install Fly CLI**:
   ```bash
   # macOS
   brew install flyctl

   # Or universal installer
   curl -L https://fly.io/install.sh | sh
   ```

2. **Sign up/login**:
   ```bash
   fly auth signup  # First time
   # OR
   fly auth login   # Existing account
   ```

### Deployment Steps

1. **Navigate to HelpfulBatBot**:
   ```bash
   cd /path/to/underworld3/HelpfulBatBot
   ```

2. **Launch the app** (first time only):
   ```bash
   fly launch --no-deploy
   # Answer the prompts:
   # - App name: uw3-helpfulbatbot (or choose your own)
   # - Region: Sydney (syd) - or closest to you
   # - Database: No
   # - Upstash Redis: No
   ```
   This creates `fly.toml` (already provided) and registers the app.

3. **Set secrets**:
   ```bash
   fly secrets set ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxx"
   ```

4. **Create persistent volume** (for repo clone):
   ```bash
   fly volumes create repo_data --size 1 --region syd
   ```

5. **Update fly.toml** to mount volume:
   Add this section to `fly.toml`:
   ```toml
   [[mounts]]
     source = "repo_data"
     destination = "/data"
   ```

6. **Deploy**:
   ```bash
   fly deploy
   ```

   This will:
   - Build the Docker image
   - Push to Fly.io registry
   - Deploy to Sydney region
   - Set up HTTPS automatically
   - Give you a URL: https://uw3-helpfulbatbot.fly.dev

7. **Set up repo sync** (one-time):
   ```bash
   # SSH into the running instance
   fly ssh console

   # Clone underworld3 into persistent volume
   cd /data
   git clone https://github.com/underworldcode/underworld3.git repo

   # Set the path
   exit
   fly secrets set BOT_REPO_PATH=/data/repo

   # Restart to pick up the change
   fly deploy
   ```

8. **Set up auto-refresh** (optional - keep repo updated):
   ```bash
   # Add a cron job to refresh the repo daily
   fly ssh console

   # Inside the container, add to crontab:
   echo "0 2 * * * cd /data/repo && git pull" | crontab -
   exit
   ```

### Verify Deployment

```bash
# Check status
fly status

# View logs
fly logs

# Test the endpoint
curl https://uw3-helpfulbatbot.fly.dev/ask \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I rebuild underworld3?", "max_context_items": 6}'
```

---

## Update GitHub Workflow

Once deployed, update `.github/workflows/repo-support-bot.yml` with your Fly.io URL:

```yaml
- name: Ask bot
  id: askbot
  uses: fjogeleit/http-request-action@v1
  with:
    url: https://uw3-helpfulbatbot.fly.dev/ask  # Your Fly.io URL
    method: "POST"
    customHeaders: |
      Content-Type: application/json
    data: |
      { "question": "${{ steps.extract.outputs.result }}", "max_context_items": 6 }
```

---

## Costs & Scaling

**Free Tier** (sufficient for low-traffic bot):
- 3 shared-CPU VMs
- 3GB storage
- 160GB transfer/month
- Auto-sleep when idle

**Estimated Monthly Cost**: $0-5 depending on usage

**Scaling Options**:
- `fly scale count 2` - Run 2 instances (redundancy)
- `fly scale memory 2048` - Increase to 2GB RAM (faster indexing)
- `fly scale vm shared-cpu-2x` - More CPU power

---

## Troubleshooting

**Bot not responding**:
```bash
fly logs --tail  # Watch live logs
```

**Out of memory**:
```bash
fly scale memory 2048  # Increase to 2GB
```

**Repo not indexed**:
```bash
fly ssh console
ls -la /data/repo  # Should see underworld3 files
python refresh_index.py  # Manually rebuild index
```

**Need to update code**:
```bash
# Just edit files in HelpfulBatBot/ and redeploy
fly deploy
```

---

## Advanced: Add Health Check Endpoint

Add to `HelpfulBat_app.py`:

```python
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "ok",
        "index_built": index_built,
        "doc_count": len(doc_store),
        "model": MODEL_NAME
    }
```

---

## Cleanup

If you want to tear everything down:

```bash
fly apps destroy uw3-helpfulbatbot
fly volumes destroy repo_data
```

---

## Next Steps

1. **Test locally first**: Make sure everything works on your machine
2. **Deploy to Fly.io**: Follow the steps above
3. **Update GitHub workflow**: Point to your Fly.io URL
4. **Add to documentation**: Create chat widget for UW3 docs (see main README)
5. **Monitor usage**: `fly dashboard` shows metrics

For questions, see the main HelpfulBat_README.md or Anthropic's Claude documentation.
