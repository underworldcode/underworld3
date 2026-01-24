# Container Strategy

Underworld3 provides two container deployment strategies for different use cases:

1. **Binder containers** - Pre-built images for mybinder.org web-based launches
2. **Command-line containers** - Lightweight images for local Docker/Podman use

## Architecture Overview

```
                    underworld3 repository
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   Dockerfile.base      Containerfile      GitHub Actions
   .optimized           (micromamba)       ┌─────────────┐
         │                  │              │binder-image │
         │                  │              │    .yml     │
         ▼                  ▼              └──────┬──────┘
   GHCR (binder)      GHCR (CLI)                  │
   ~3.4GB slim        ~2GB                        ▼
         │                               uw3-binder-launcher
         │                               (auto-updated)
         ▼                                       │
   mybinder.org                                  ▼
                                          mybinder.org
```

## Binder Containers

### Purpose

Binder containers are optimized for [mybinder.org](https://mybinder.org) launches, providing:

- Pre-compiled Underworld3 with all dependencies
- Jupyter environment ready for notebooks
- Fast launch times (no build required on mybinder.org)

### Key Files

| File | Location | Purpose |
|------|----------|---------|
| `Dockerfile.base.optimized` | `container/` | Primary binder Dockerfile (~3.4GB) |
| `binder-image.yml` | `.github/workflows/` | GitHub Actions build workflow |
| `update-image.yml` | `uw3-binder-launcher/.github/workflows/` | Auto-update launcher |

### Image Registry

Images are pushed to GitHub Container Registry (GHCR):

```
ghcr.io/underworldcode/uw3-base:<branch>-slim
ghcr.io/underworldcode/uw3-base:latest-slim
```

Branch-specific tags (`main-slim`, `development-slim`) enable testing different versions.

### Build Triggers

The `binder-image.yml` workflow triggers on:

- Changes to `container/Dockerfile.base.optimized`
- Changes to `pixi.toml` or `pixi.lock`
- Changes to Cython files (`src/**/*.pyx`, `src/**/*.c`)
- Changes to `setup.py`
- Manual dispatch (with optional force rebuild)

```{note}
Pure Python changes (`.py` files) don't trigger rebuilds because they're pulled at runtime via `nbgitpuller`.
```

### Layer Size Constraints

mybinder.org enforces ~1GB layer size limits. The optimized Dockerfile splits the `lib` directory across multiple layers to stay under this limit.

### Launcher Repository

The [uw3-binder-launcher](https://github.com/underworldcode/uw3-binder-launcher) repository serves as the mybinder.org entry point:

**Structure**:
```
uw3-binder-launcher/
├── .binder/
│   └── Dockerfile       # FROM ghcr.io/underworldcode/uw3-base:<branch>-slim
├── .github/workflows/
│   └── update-image.yml # Auto-updates Dockerfile on new builds
└── README.md            # Badge links and usage instructions
```

**Branch Mapping**:

| Launcher Branch | UW3 Branch | Binder URL |
|-----------------|------------|------------|
| `main` | `main` | `mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/main` |
| `development` | `development` | `mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/development` |

### Automation Pipeline

When code is pushed to a tracked branch:

1. **Build**: `binder-image.yml` builds and pushes to GHCR
2. **Dispatch**: Sends `repository_dispatch` event to launcher repo
3. **Update**: Launcher's `update-image.yml` updates the Dockerfile
4. **Ready**: mybinder.org uses updated image on next launch

```{tip}
The `LAUNCHER_PAT` secret (Personal Access Token with `repo` scope) enables cross-repository dispatch.
```

### Using nbgitpuller

Any repository can launch on mybinder.org using the pre-built image via nbgitpuller:

```
https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FYOUR_ORG%252FYOUR_REPO%26urlpath%3Dlab%252Ftree%252FYOUR_REPO%252Fpath%252Fto%252Fnotebook.ipynb
```

Use the `scripts/binder_wizard.py` script to generate these URLs.

## Command-Line Containers

### Purpose

Command-line containers provide a lightweight option for users who want to run Underworld3 locally without installing dependencies:

- Micromamba-based (smaller image size)
- Jupyter server included
- Volume mounting for data transfer

### Key Files

| File | Location | Purpose |
|------|----------|---------|
| `Containerfile` | `container/` | Micromamba-based image (~2GB) |
| `launch-container.sh` | `container/` | Podman launch script |
| `docker-image.yml` | `.github/workflows/` | GHCR build workflow |

### Building Locally

```bash
# With Podman
podman build . --rm \
    --format docker \
    -f ./container/Containerfile \
    -t underworldcode/underworld3:local

# With Docker
docker build -f container/Containerfile -t underworldcode/underworld3:local .
```

### Running

**Using the launch script** (recommended for Podman):

```bash
./container/launch-container.sh
```

This script:
- Creates `$HOME/uw_space` for file transfer
- Maps it to `/home/mambauser/host` in the container
- Runs Jupyter on port 10000: `http://localhost:10000`
- Handles rootless Podman UID/GID mapping

**Manual Docker run**:

```bash
docker run -it --rm -p 8888:8888 ghcr.io/underworldcode/underworld3:development
```

### Rootless Podman

The launch script includes UID/GID mapping for rootless Podman:

```bash
podman run -it --rm \
  -p 10000:8888 \
  --uidmap $uid:0:1 \
  --uidmap 0:1:$uid \
  # ... additional mappings
  -v "${HOME}/uw_space":/home/mambauser/host \
  ghcr.io/underworldcode/underworld3:development
```

```{warning}
Do NOT run the launch script with `sudo`. Rootless Podman requires the executing user to be non-root for proper namespace mapping.
```

### Image Registry

Command-line images are pushed to GHCR (same registry as binder images):

```
ghcr.io/underworldcode/underworld3:<branch>
ghcr.io/underworldcode/underworld3:latest
```

Builds trigger on pushes to `main` and `development` branches when container-related files change. Can also be triggered manually via workflow_dispatch.

## Comparison

| Aspect | Binder | Command-Line |
|--------|--------|--------------|
| **Dockerfile** | `Dockerfile.base.optimized` | `Containerfile` |
| **Base** | Ubuntu + Pixi | Micromamba |
| **Size** | ~3.4GB (slim) | ~2GB |
| **Registry** | GHCR | GHCR |
| **Use case** | mybinder.org | Local `docker run` |
| **Workflow** | `binder-image.yml` | `docker-image.yml` |
| **Automation** | Full (build + launcher update) | Build only |

## Architecture Constraints

### Platform Support

Currently only `linux/amd64` is built because `vtk-osmesa` (required for headless rendering) is not available for ARM architectures.

### When Rebuilds Are Required

| Change Type | Rebuild Required? | Reason |
|-------------|-------------------|--------|
| `.py` files | No | Runtime pull works |
| `.pyx` files | **Yes** | Cython needs recompile |
| `pixi.toml` | **Yes** | Dependencies changed |
| `setup.py` | **Yes** | Build config changed |
| Notebooks | No | Runtime pull works |
| Documentation | No | Not in container |

## Troubleshooting

### GHCR Permission Errors

If you see `permission_denied: write_package`:

1. Go to the package settings on GHCR
2. Under "Manage Actions access", add the repository with write permission
3. Ensure the repository is linked to the package

### repository_dispatch Not Triggering

The `update-image.yml` workflow must be on the **default branch** (usually `main`) of the launcher repository to receive `repository_dispatch` events.

### mybinder.org Layer Size Errors

If builds fail with layer size errors:

1. Check the `Dockerfile.base.optimized` layer splitting
2. Ensure no single layer exceeds ~1GB
3. Consider further splitting large directories

## Related Resources

- [mybinder.org documentation](https://mybinder.readthedocs.io/)
- [Container stacks with Podman](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html#using-the-podman-cli)
- [Micromamba images](https://micromamba-docker.readthedocs.io/en/latest/quick_start.html)
- GitHub repositories:
  - [underworld3](https://github.com/underworldcode/underworld3)
  - [uw3-binder-launcher](https://github.com/underworldcode/uw3-binder-launcher)
