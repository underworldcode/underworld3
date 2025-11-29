# CuckooBot Dockerfile
# Self-contained deployment for Underworld3 support bot

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for sentence-transformers)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Cuckoo_app.py app.py
COPY Cuckoo_refreshment.py refresh_index.py

# Clone underworld3 repo for indexing (will be done at runtime via volume or git)
# This is handled by BOT_REPO_PATH environment variable

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
