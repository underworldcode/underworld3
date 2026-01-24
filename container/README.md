# Container Files

This directory contains all Dockerfiles for Underworld3, supporting two use cases:

1. **Command-line containers** - For users who want to run UW3 locally without installation
2. **Binder containers** - For mybinder.org web-based launches

Both containers are hosted on **GitHub Container Registry (GHCR)** using the
repository's built-in `GITHUB_TOKEN` for authentication (no external secrets required).

---

## Command-Line Container (Containerfile)

Lightweight container for local command-line use with Docker or Podman.

| File | Purpose |
|------|---------|
| `Containerfile` | Micromamba-based image for local use |
| `launch-container.sh` | Podman launch script with rootless support |

### Pulling Pre-Built Images

```bash
# Pull the latest development image
docker pull ghcr.io/underworldcode/underworld3:development

# Or a specific branch
docker pull ghcr.io/underworldcode/underworld3:main
```

### Running

```bash
# Run the pre-built image
docker run -it --rm -p 8888:8888 ghcr.io/underworldcode/underworld3:development

# Using the launch script (Podman, recommended for local builds)
./container/launch-container.sh
```

The launch script:
- Maps `$HOME/uw_space` into the container for file transfer
- Runs Jupyter on port 10000 (http://localhost:10000)
- Handles rootless Podman UID/GID mapping

### Building Locally

```bash
# From repository root (requires Docker or Podman)
podman build . --rm \
    --format docker \
    -f ./container/Containerfile \
    -t underworld3:local

# Or with Docker
docker build -f container/Containerfile -t underworld3:local .
```

### GitHub Actions

The `docker-image.yml` workflow automatically builds and pushes to GHCR when:
- `container/Containerfile` changes
- `environment.yaml` changes
- Cython files (`.pyx`) or C files change
- `setup.py` or `pyproject.toml` changes

Can also be triggered manually via workflow_dispatch.

### Architecture

At present only amd64 architecture is built, because vtk-osmesa isn't available for arm by default.

Useful links:
- [Container stacks with Podman](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html#using-the-podman-cli)
- [Micromamba images](https://micromamba-docker.readthedocs.io/en/latest/quick_start.html#quick-start)
- [PyVista containers](https://github.com/pyvista/pyvista/tree/main/docker)

---

## Binder Container (Dockerfile.base.optimized)

Full-featured container optimized for mybinder.org launches.

| File | Purpose |
|------|---------|
| `Dockerfile.base.optimized` | **Primary** - Optimized slim image (~3.4GB) |
| `Dockerfile.base` | Full image with dev tools (~8.3GB) |
| `Dockerfile.deps` | Dependencies-only layer (two-stage builds) |
| `Dockerfile.branch` | Branch-specific layer (two-stage builds) |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  This repo: underworld3                                             │
│                                                                     │
│  container/Dockerfile.base.optimized                                │
│      ↓ (GitHub Actions: binder-image.yml)                          │
│  ghcr.io/underworldcode/uw3-base:<branch>-slim                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Launcher repo: uw3-binder-launcher                                 │
│                                                                     │
│  .binder/Dockerfile:                                                │
│      FROM ghcr.io/underworldcode/uw3-base:<branch>-slim            │
│      ENV UW3_BRANCH=<branch>                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  mybinder.org                                                       │
│                                                                     │
│  Uses launcher repo → pulls pre-built image → fast launches        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Building Locally

```bash
# From repository root
docker build --platform linux/amd64 \
  -f container/Dockerfile.base.optimized \
  -t ghcr.io/underworldcode/uw3-base:test-slim .

# Test locally
docker run --rm -p 8888:8888 ghcr.io/underworldcode/uw3-base:test-slim
```

### Layer Size Constraints

mybinder.org has an ~1GB layer size limit. The optimized Dockerfile splits the `lib` directory into multiple layers to stay under this limit. See `docs/developer/guides/BINDER_CONTAINER_SETUP.md` for details.

### GitHub Actions

The `binder-image.yml` workflow automatically builds and pushes images when:
- `container/Dockerfile.base.optimized` changes
- `pixi.toml` or `pixi.lock` changes
- Cython files (`.pyx`) change
- `setup.py` changes

It also triggers the launcher repo to update its image reference.

---

## Comparison

| Aspect | Command-Line | Binder |
|--------|--------------|--------|
| **File** | `Containerfile` | `Dockerfile.base.optimized` |
| **Base** | micromamba | Ubuntu + Pixi |
| **Size** | ~2GB | ~3.4GB (slim) |
| **Registry** | GHCR | GHCR |
| **Image** | `ghcr.io/underworldcode/underworld3:<branch>` | `ghcr.io/underworldcode/uw3-base:<branch>-slim` |
| **Use case** | Local `docker run` | mybinder.org |
| **Workflow** | `docker-image.yml` | `binder-image.yml` |

## Related

- **Binder setup docs**: `docs/developer/guides/BINDER_CONTAINER_SETUP.md`
- **Launcher repo**: https://github.com/underworldcode/uw3-binder-launcher
- **Badge generator**: `scripts/binder_wizard.py`
