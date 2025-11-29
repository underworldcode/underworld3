<!-- filename: docs/BOT_README.md -->
# GitHub Repo Support Bot

## Planning

Below is a minimal, production-ready setup for a GitHub-native support bot that: listens to Issues and Discussions, retrieves answers grounded in your repo + README/docs, replies with runnable code blocks, and always includes file/line citations. It uses Python (FastAPI + FAISS) and a GitHub App via Actions. You can deploy it on any server.


  1. Index your GitHub repo: clone main, chunk files, embed with SentenceTransformers, store in FAISS.

  2. Retrieval + answering: FastAPI endpoint /ask that retrieves top-k chunks and calls your LLM with a strict “citations required” system prompt.

  3. GitHub integration: a workflow triggers on issues/comments, calls your bot, posts a reply with citations.

  4. Guardrails: deny answers without citations; rate-limit; never run arbitrary code; allow “I don’t know.”

  5. Config: set environment variables for repo path, GitHub blob base URL, and your LLM endpoint/key.



## What it does
- Replies to Issues/Discussions with grounded answers and runnable code.
- Cites file paths + line ranges so users can verify.
- Says “I don’t know” when context is insufficient.

## Deploy
- Host FastAPI (`app.py`), set env:
  - BOT_REPO_PATH=/srv/repos/ORG/REPO  (keep updated via cron or CI)
  - BOT_BASE_URL=https://github.com/ORG/REPO/blob/main
  - BOT_LLM_ENDPOINT=YOUR_ENDPOINT
  - BOT_LLM_API_KEY=YOUR_KEY
- Expose POST /ask
- Add workflow `.github/workflows/repo-support-bot.yml`

## Security
- Don’t log secrets.
- Never execute user-provided code on host.
- Rate-limit replies; add basic abuse detection.

## Troubleshooting
- If no citations in replies: check BOT_BASE_URL and index completeness.
- If irrelevant context: tune chunk size/overlap; consider keyword fallback.
