# Binder Container Files

This directory contains Dockerfiles for building the **mybinder.org** base image. These are separate from the command-line Docker setup in `docs/developer/container/`.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.base.optimized` | **Primary** - Optimized slim image for binder (~3.4GB) |
| `Dockerfile.base` | Full image with dev tools (~8.3GB) |
| `Dockerfile.deps` | Dependencies-only layer (for two-stage builds) |
| `Dockerfile.branch` | Branch-specific layer (for two-stage builds) |

## Architecture

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

## Building Locally

```bash
# From repository root
docker build --platform linux/amd64 \
  -f container/Dockerfile.base.optimized \
  -t ghcr.io/underworldcode/uw3-base:test-slim .

# Test locally
docker run --rm -p 8888:8888 ghcr.io/underworldcode/uw3-base:test-slim
```

## Layer Size Constraints

mybinder.org has an ~1GB layer size limit. The optimized Dockerfile splits the `lib` directory into multiple layers to stay under this limit. See `docs/developer/BINDER_CONTAINER_SETUP.md` for details.

## GitHub Actions

The `binder-image.yml` workflow automatically builds and pushes images when:
- `container/Dockerfile.base.optimized` changes
- `pixi.toml` or `pixi.lock` changes
- Cython files (`.pyx`) change
- `setup.py` changes

It also triggers the launcher repo to update its image reference.

## Related

- **Binder setup docs**: `docs/developer/BINDER_CONTAINER_SETUP.md`
- **Launcher repo**: https://github.com/underworldcode/uw3-binder-launcher
- **Command-line Docker**: `docs/developer/container/` (separate from binder)
- **Badge generator**: `scripts/binder_wizard.py`
