# filename: app.py
import os
import json
import uvicorn
from typing import List, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path, PurePosixPath
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Embeddings & retrieval
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic  # Claude API

# Env vars (configure in your host)
# BOT_REPO_PATH: local path to a checkout of your GitHub repo (kept updated via cron/CI)
# BOT_BASE_URL: GitHub blob base, e.g. https://github.com/ORG/REPO/blob/main
# ANTHROPIC_API_KEY: your Anthropic API key for Claude
# BOT_MAX_FILE_SIZE: optional, default 200_000 chars
# BOT_ALLOWED_EXTS: optional, comma-separated (default typical code/doc exts)
# CLAUDE_MODEL: optional, default claude-3-5-sonnet-20241022

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")


class Query(BaseModel):
    question: str
    max_context_items: int = 6


class IndexedDoc(BaseModel):
    doc_id: int
    path: str
    start_line: int
    end_line: int
    text: str


class BotResponse(BaseModel):
    answer: str
    citations: List[str]
    used_files: List[str]
    confidence: float


app = FastAPI(title="GitHub Repo Support Bot")

index_built = False
faiss_index = None
doc_store: List[IndexedDoc] = []
embedder: Optional[SentenceTransformer] = None


def allowed_exts() -> set:
    exts_env = os.environ.get("BOT_ALLOWED_EXTS")
    if exts_env:
        return set(e.strip().lower() for e in exts_env.split(",") if e.strip())
    return {
        ".py",
        ".md",
        ".txt",
        ".ipynb",  # Added Jupyter notebook support
        ".c",
        ".h",
        ".hpp",
        ".cc",
        ".cpp",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".sh",
        ".bash",
        ".zsh",
        ".typ",
    }


def should_include_file(rel_path: str) -> bool:
    """
    Check if file should be indexed based on path patterns.

    Uses BOT_INCLUDE_PATHS and BOT_EXCLUDE_PATHS environment variables.
    If not set, uses sensible defaults for user-facing UW3 content.
    """
    include_env = os.environ.get("BOT_INCLUDE_PATHS")
    exclude_env = os.environ.get("BOT_EXCLUDE_PATHS")

    # Convert to PurePosixPath for pattern matching (works with ** patterns)
    path = PurePosixPath(rel_path)

    # Default: index user-facing content only
    default_includes = [
        "docs/beginner/tutorials/*.ipynb",
        "docs/beginner/tutorials/*.md",
        "docs/beginner/*.md",
        "docs/advanced/**/*.ipynb",
        "docs/advanced/**/*.md",
        "examples/*.ipynb",
        "examples/*.py",
        "tests/test_0[0-6]*.py",  # A/B grade tests only
        "README.md",
        "CLAUDE.md",
        "docs/*.md",
    ]

    # Default: exclude internal implementation details
    default_excludes = [
        "src/**/*",                # Source code internals
        "docs/developer/**/*",     # Developer docs
        "docs/planning/**/*",      # Planning documents in docs
        "planning/**/*",           # Planning documents
        "SESSION-SUMMARY-*.md",    # Session summaries
        "tests/test_[7-9]*.py",    # C/D grade tests
        "tests/test_1*.py",        # Complex tests
        ".git/**/*",               # Git metadata
        "**/__pycache__/**/*",     # Python cache
        "build/**/*",              # Build artifacts
        ".github/**/*",            # GitHub workflows
        ".ipynb_checkpoints/**/*", # Notebook checkpoints
        ".pytest_cache/**/*",      # Pytest cache
        ".quarto/**/*",            # Quarto build files
        "_freeze/**/*",            # Quarto frozen files
        "docs/.quarto/**/*",       # Quarto docs cache
        "docs/_freeze/**/*",       # Quarto docs frozen
        "HelpfulBatBot/**/*",          # HelpfulBatBot directory itself
        "temp_tests_deletable/**/*",  # Temporary test files
        "conda/**/*",              # Conda build files
        "publications/**/*",       # Publications (not user docs)
        "docs_legacy/**/*",        # Legacy documentation
        "**/output/**/*",          # Output directories
        "**/.claude/**/*",         # Claude cache
    ]

    # Use env vars if provided, otherwise use defaults
    includes = default_includes
    excludes = default_excludes

    if include_env:
        includes = [p.strip() for p in include_env.split(",") if p.strip()]
    if exclude_env:
        excludes.extend([p.strip() for p in exclude_env.split(",") if p.strip()])

    # Check excludes first (they take priority)
    for pattern in excludes:
        if path.match(pattern):
            return False

    # Check includes
    for pattern in includes:
        if path.match(pattern):
            return True

    # If we're using includes (default or env), reject files that don't match
    # Only allow through if there are NO include patterns defined
    return False


def extract_notebook_text(nb_path: Path) -> str:
    """
    Extract text content from Jupyter notebook (.ipynb) file.

    Extracts both markdown cells and code cells for indexing.
    """
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        text_parts = []

        # Add notebook title/path as context
        text_parts.append(f"# Jupyter Notebook: {nb_path.name}\n")

        for i, cell in enumerate(nb.get('cells', []), 1):
            cell_type = cell.get('cell_type')
            source = cell.get('source', [])

            # source can be a list of lines or a single string
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source

            if not content.strip():
                continue

            if cell_type == 'markdown':
                text_parts.append(f"## Cell {i} (Markdown)\n{content}\n")
            elif cell_type == 'code':
                text_parts.append(f"## Cell {i} (Code)\n```python\n{content}\n```\n")

        return '\n\n'.join(text_parts)

    except Exception as e:
        # If we can't parse the notebook, return empty string
        return ""


def load_files(repo_path: str) -> List[Tuple[str, str]]:
    """
    Load files from repository for indexing.

    Supports:
    - Extension-based filtering (BOT_ALLOWED_EXTS)
    - Path-based filtering (BOT_INCLUDE_PATHS, BOT_EXCLUDE_PATHS)
    - Jupyter notebook extraction (.ipynb)
    - Size limiting (BOT_MAX_FILE_SIZE)
    """
    max_size = int(os.environ.get("BOT_MAX_FILE_SIZE", "200000"))
    exts = allowed_exts()
    files = []
    root = Path(repo_path)

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        rel_path = str(p.relative_to(root))

        # Path-based filtering (includes and excludes)
        if not should_include_file(rel_path):
            continue

        # Extension filtering
        if p.suffix.lower() not in exts:
            continue

        try:
            # Special handling for Jupyter notebooks
            if p.suffix.lower() == '.ipynb':
                content = extract_notebook_text(p)
            else:
                content = p.read_text(encoding="utf-8", errors="ignore")

            # Skip if empty or too large
            if not content or len(content) > max_size:
                continue

            files.append((rel_path, content))

        except Exception:
            continue

    return files


def chunk_text(path: str, text: str, max_chars: int = 2000, overlap: int = 200) -> List[IndexedDoc]:
    lines = text.splitlines()
    chunks = []
    start = 0
    base_id = len(doc_store)
    while start < len(lines):
        acc = []
        acc_len = 0
        i = start
        while i < len(lines) and acc_len + len(lines[i]) + 1 <= max_chars:
            acc.append(lines[i])
            acc_len += len(lines[i]) + 1
            i += 1
        chunk = "\n".join(acc)
        chunks.append(
            IndexedDoc(
                doc_id=base_id + len(chunks),
                path=path,
                start_line=start + 1,
                end_line=i,
                text=chunk,
            )
        )
        start = max(i - overlap, start + 1)
        if start >= i:
            start = i
    return chunks


def ensure_index():
    global index_built, faiss_index, doc_store, embedder
    if index_built:
        return
    repo_path = os.environ.get("BOT_REPO_PATH")
    if not repo_path:
        raise RuntimeError("BOT_REPO_PATH not set")
    files = load_files(repo_path)
    embedder = SentenceTransformer(MODEL_NAME)
    embeddings = []
    docs = []
    for path, content in files:
        for ch in chunk_text(path, content):
            docs.append(ch)
            emb = embedder.encode(ch.text, normalize_embeddings=True).astype(np.float32)
            embeddings.append(emb)
    if not embeddings:
        raise RuntimeError("No documents indexed")
    mat = np.vstack(embeddings)
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)  # cosine via normalized embeddings
    faiss_index.add(mat)
    doc_store = docs
    index_built = True


def retrieve(question: str, k: int) -> List[IndexedDoc]:
    ensure_index()
    q_emb = embedder.encode(question, normalize_embeddings=True).astype(np.float32)
    D, I = faiss_index.search(q_emb.reshape(1, -1), k)
    return [doc_store[idx] for idx in I[0] if idx != -1]


def linkify(path: str, start_line: int, end_line: int) -> str:
    base = os.environ.get("BOT_BASE_URL")
    if not base:
        return f"{path}#L{start_line}-L{end_line}"
    return f"{base}/{path}#L{start_line}-L{end_line}"


def build_system_prompt() -> str:
    return (
        "You are an expert assistant for Underworld3, a geodynamics modeling framework.\n"
        "- You understand PETSc, parallel computing, finite element methods, and computational geodynamics.\n"
        "- Answer ONLY using the provided repository context.\n"
        "- If context is insufficient, acknowledge limitations and suggest where to look.\n"
        "- Provide concise, correct, runnable code examples with proper imports.\n"
        "- ALWAYS cite file paths and line ranges (format: `file.py:123-145`).\n"
        "- For solver questions, mention PETSc compatibility requirements.\n"
        "- For parallel safety, reference patterns in CLAUDE.md (use uw.pprint(), uw.selective_ranks()).\n"
        "- Never promise features or roadmap items not explicitly in the code.\n"
        "\n"
        "Key priorities:\n"
        "1. Solver stability is paramount (never suggest changes to core solvers)\n"
        "2. Always rebuild after source changes: `pixi run underworld-build`\n"
        "3. Parallel safety is critical in all examples"
    )


def format_context(ctx: List[IndexedDoc]) -> str:
    return "\n\n".join(f"[{d.path}:{d.start_line}-{d.end_line}]\n{d.text}" for d in ctx)


def call_llm_with_caching(system_prompt: str, user_prompt: str, context: str) -> str:
    """
    Call Claude with prompt caching for cost savings.

    The context is cached, so repeated queries with similar context
    cost 90% less after the first query.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "I don't have Claude configured. Set ANTHROPIC_API_KEY environment variable."

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0.2,
            system=[
                {
                    "type": "text",
                    "text": system_prompt
                },
                {
                    "type": "text",
                    "text": f"Repository context (this is cached for efficiency):\n\n{context}",
                    "cache_control": {"type": "ephemeral"}  # Cache this part!
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return message.content[0].text

    except anthropic.APIError as e:
        return f"Claude API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error calling Claude: {str(e)}"


def enforce_citations(answer_md: str, ctx: List[IndexedDoc]) -> Tuple[str, List[str], List[str]]:
    used = sorted({d.path for d in ctx if d.path in answer_md})
    citations = []
    for d in ctx:
        if d.path in answer_md:
            citations.append(linkify(d.path, d.start_line, d.end_line))
    if not citations:
        return (
            "I don’t have enough repo context to answer confidently. "
            "Please share the relevant file path or snippet.",
            [],
            [],
        )
    return (answer_md, citations, used)


@app.post("/ask", response_model=BotResponse)
def ask(q: Query):
    ctx = retrieve(q.question, k=q.max_context_items)
    system_prompt = build_system_prompt()
    context_text = format_context(ctx)
    user_prompt = (
        f"Question: {q.question}\n\n"
        "Provide a clear markdown answer with code examples if applicable. "
        "Include citations to specific files and line ranges."
    )
    raw = call_llm_with_caching(system_prompt, user_prompt, context_text)
    answer, citations, used_files = enforce_citations(raw, ctx)
    confidence = 0.5 if "don't have" in answer or "Claude" in answer and "error" in answer else 0.8
    return BotResponse(
        answer=answer, citations=citations, used_files=used_files, confidence=confidence
    )


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "ok",
        "index_built": index_built,
        "doc_count": len(doc_store) if doc_store else 0,
        "embedding_model": MODEL_NAME,
        "claude_model": CLAUDE_MODEL
    }


def find_available_port(start_port=8001, max_attempts=10):
    """
    Find an available port starting from start_port.

    Args:
        start_port: Port to start searching from (default: 8001)
        max_attempts: Maximum number of ports to try (default: 10)

    Returns:
        int: Available port number

    Raises:
        RuntimeError: If no available port found in range
    """
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port+max_attempts}")


def write_port_file(port, port_file="bot.port"):
    """
    Write the port number to a file so clients can find the bot.

    Args:
        port: Port number to write
        port_file: File to write port to (default: bot.port)
    """
    port_path = Path(__file__).parent / port_file
    port_path.write_text(str(port))
    print(f"📝 Port {port} written to {port_path}")


if __name__ == "__main__":
    # Find available port
    port = find_available_port(8001)
    print(f"🚀 Starting HelpfulBatBot on port {port}")

    # Write port to file for clients
    write_port_file(port)

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)
