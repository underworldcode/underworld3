# Binder Container Setup for Underworld3

**Date**: 2025-01-14
**Status**: Implemented and working

## Overview

Underworld3 uses a **pre-built container strategy** with a **separate launcher repository** for mybinder.org launches. This two-repository approach provides:
- Fast startup times (no container build by mybinder.org)
- Stable caching (launcher repo changes rarely)
- Branch selection (different launcher branches pull different underworld3 branches)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GHCR (GitHub Container Registry)            │
│                                                                     │
│   ghcr.io/underworldcode/uw3-base:2025.01                          │
│   ├── Ubuntu 24.04 base                                            │
│   ├── Pixi package manager                                         │
│   ├── Full runtime environment (PETSc, gmsh, pyvista, etc.)        │
│   ├── Underworld3 built and installed                              │
│   ├── Jupyter kernel configured                                    │
│   ├── Start script with UW3_BRANCH support                         │
│   └── ENTRYPOINT configured                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ docker pull (cached at node level)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         mybinder.org                                │
│                                                                     │
│   1. User launches via uw3-binder-launcher repo                    │
│   2. Reads .binder/Dockerfile (FROM + ENV UW3_BRANCH)              │
│   3. Caches by launcher repo commit (rarely changes)               │
│   4. Pulls pre-built image from GHCR                               │
│   5. Runs container with start script                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ on container start
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Start Script                                │
│                                                                     │
│   1. git fetch/checkout/pull ${UW3_BRANCH}                         │
│   2. pixi run -e runtime build  (~30 seconds)                      │
│   3. Copy tutorials to workspace                                   │
│   4. exec pixi run -e runtime jupyter lab                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

### Launcher Repository (`uw3-binder-launcher`)

A minimal repository that exists solely to provide stable mybinder.org caching:

**Location**: https://github.com/underworldcode/uw3-binder-launcher

**Branches**:
| Branch | UW3_BRANCH | Purpose |
|--------|------------|---------|
| `main` | `main` | Stable release |
| `uw3-release-candidate` | `uw3-release-candidate` | Release candidate testing |
| `development` | `development` | Development branch |

**Contents** (each branch):
```
uw3-binder-launcher/
├── .binder/
│   └── Dockerfile      # FROM + ENV UW3_BRANCH=<branch>
└── README.md           # Launch badges and documentation
```

### Main Repository (`underworld3`)

The actual Underworld3 codebase:

**Location**: https://github.com/underworldcode/underworld3

**Key files**:
| File | Purpose |
|------|---------|
| `container/Dockerfile.base` | Recipe for building the base image |
| `pixi.toml` | Environment definition |
| `docs/beginner/tutorials/` | Tutorial notebooks |

## Key Benefits

1. **Fast Launches**: No Docker build required - just pull cached image and run
2. **Stable Caching**: mybinder.org caches by launcher repo commit hash; launcher rarely changes
3. **Always Current**: Start script pulls latest underworld3 code on each launch
4. **Branch Selection**: Different launcher branches pull different underworld3 branches
5. **Simple Maintenance**: Push code to underworld3; users get latest automatically

## Image Variants

| Tag | Size | Purpose |
|-----|------|---------|
| `2025.01-slim` | ~1.4GB | **Recommended** - Optimized for binder launches |
| `2025.01` | ~8.3GB | Full image with headers/toolchain (for development) |

The slim image removes components not needed at runtime:
- Git history (`.git` directory)
- Legacy documentation (`docs_legacy`)
- C/C++ headers (`include/`)
- Compiler toolchain (`x86_64-conda-linux-gnu/`)
- Package metadata (`conda-meta/`)

## Launch URLs

### Quick Launch Badges

| Branch | Launch |
|--------|--------|
| `main` | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/main) |
| `uw3-release-candidate` | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/uw3-release-candidate) |
| `development` | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/development) |

### URL Format

```
https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/<branch>
```

### Badge Markdown

```markdown
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/main)
```

## Launcher Dockerfile

Each branch in `uw3-binder-launcher` has a minimal `.binder/Dockerfile`:

```dockerfile
# Binder launcher for Underworld3 - <branch> branch
# Uses pre-built base image from GHCR
FROM ghcr.io/underworldcode/uw3-base:2025.01-slim

# Pull from <branch> branch of underworld3
ENV UW3_BRANCH=<branch>
```

The `UW3_BRANCH` environment variable tells the start script which branch to pull.

## Base Image Contents

The base image (built from `container/Dockerfile.base.optimized` for slim variant) includes:

### System Dependencies
- Ubuntu 24.04
- OpenGL/Mesa libraries for visualization (including OSMesa for software rendering)
- X11 libraries for gmsh
- Git, curl, ca-certificates

### Python Environment (via Pixi)
- Python with full scientific stack
- PETSc and petsc4py
- gmsh for meshing
- pyvista for visualization
- JupyterLab
- All underworld3 dependencies

### Underworld3
- Cloned from repository
- Built and installed in pixi environment
- Jupyter kernel registered as "Underworld3"

### Start Script
Embedded at `/home/jovyan/start`:

```bash
#!/bin/bash
cd /home/jovyan/underworld3
git fetch origin ${UW3_BRANCH:-uw3-release-candidate} 2>/dev/null || true
git checkout ${UW3_BRANCH:-uw3-release-candidate} 2>/dev/null || true
git pull origin ${UW3_BRANCH:-uw3-release-candidate} 2>/dev/null || true
pixi run -e runtime build 2>/dev/null || true
cp -r docs/beginner/tutorials/* /home/jovyan/Tutorials/ 2>/dev/null || true
exec pixi run -e runtime "$@"
```

The script:
1. Uses `UW3_BRANCH` env var (defaults to `uw3-release-candidate`)
2. Fetches, checkouts, and pulls the specified branch
3. Rebuilds underworld3 (~30 seconds)
4. Copies tutorials to workspace
5. Launches jupyter (or whatever command was passed)

### Binder Configuration
- User: `jovyan` (UID 1000) - mybinder.org standard
- ENTRYPOINT: `/home/jovyan/start`
- CMD: `jupyter lab --ip=0.0.0.0 --no-browser`
- Tutorials pre-copied to `/home/jovyan/Tutorials/`

## Maintenance

### Updating Code (No Image Rebuild Needed)

For routine code changes to underworld3:
1. Push changes to the appropriate underworld3 branch
2. Users launching binder will automatically get latest code
3. The start script runs `git pull` and rebuilds on each launch

**No changes to launcher repo needed!**

### Adding a New Branch

To add a new underworld3 branch to binder:

1. **Create branch in launcher repo**:
   ```bash
   cd uw3-binder-launcher
   git checkout -b <new-branch>
   ```

2. **Edit `.binder/Dockerfile`**:
   ```dockerfile
   FROM ghcr.io/underworldcode/uw3-base:2025.01
   ENV UW3_BRANCH=<underworld3-branch-name>
   ```

3. **Push**:
   ```bash
   git push origin <new-branch>
   ```

4. **Update README** with new launch badge

### Two-Layer Build Architecture (Recommended)

For faster builds and better caching, use the two-layer architecture:

**Layer 1: Dependencies image** (`uw3-deps`) - Rarely changes, caches well
```bash
# Build dependencies-only image (do this when pixi.toml changes)
docker build --platform linux/amd64 \
  -f container/Dockerfile.deps \
  -t ghcr.io/underworldcode/uw3-deps:2025.01 .

docker push ghcr.io/underworldcode/uw3-deps:2025.01
```

**Layer 2: Branch-specific image** (`uw3-base`) - Quick rebuilds for code changes
```bash
# Build branch-specific image (do this when code changes)
docker build --platform linux/amd64 \
  --build-arg UW3_BRANCH=uw3-release-candidate \
  --build-arg DEPS_IMAGE=ghcr.io/underworldcode/uw3-deps:2025.01 \
  -f container/Dockerfile.branch \
  -t ghcr.io/underworldcode/uw3-base:2025.01-slim .

docker push ghcr.io/underworldcode/uw3-base:2025.01-slim
```

**Benefits**:
- Dependencies layer rarely changes → stays cached on binder nodes
- Branch rebuilds only need to clone code and pip install → much faster
- Multiple branches can share the same deps image

### Updating Dependencies (Image Rebuild Required)

When dependencies change (new packages, version updates):

1. **Update pixi.toml** with new dependencies

2. **Rebuild deps image first**:
   ```bash
   docker build --platform linux/amd64 \
     -f container/Dockerfile.deps \
     -t ghcr.io/underworldcode/uw3-deps:YYYY.MM .
   docker push ghcr.io/underworldcode/uw3-deps:YYYY.MM
   ```

3. **Rebuild branch image** (or use single-file Dockerfile.base.optimized):
   ```bash
   docker build --platform linux/amd64 \
     -f container/Dockerfile.base.optimized \
     -t ghcr.io/underworldcode/uw3-base:YYYY.MM-slim .
   ```

3. **Test locally**:
   ```bash
   docker run --rm -p 8888:8888 ghcr.io/underworldcode/uw3-base:YYYY.MM-slim
   ```

4. **Push to GHCR**:
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
   docker push ghcr.io/underworldcode/uw3-base:YYYY.MM-slim
   ```

5. **Update all launcher branches** with new tag:
   ```bash
   cd uw3-binder-launcher
   for branch in main uw3-release-candidate development; do
     git checkout $branch
     # Edit .binder/Dockerfile with new tag (use -slim for binder)
     git commit -am "Update to uw3-base:YYYY.MM-slim"
     git push origin $branch
   done
   ```

### GHCR Permissions

The GHCR package must be **public** for mybinder.org to pull it without authentication.

To make public:
1. Go to https://github.com/orgs/underworldcode/packages/container/uw3-base/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Public"

## Troubleshooting

### "401 Unauthorized" on mybinder.org

The GHCR image is private. Make it public (see above).

### Platform mismatch (linux-aarch64)

Building on Apple Silicon defaults to ARM. Use `--platform linux/amd64`:
```bash
docker build --platform linux/amd64 -f container/Dockerfile.base ...
```

### Slow launches

If mybinder.org seems slow:
1. First launch after cache expiry will pull the ~1.4GB slim image (much faster than full image)
2. Subsequent launches should be faster (image cached at node level)
3. The ~30 second `pixi run build` happens every launch

### Kernel not found

Notebooks must use the `python3` kernel name, not `pixi-kernel-python3`.
Tutorial notebooks have been normalized to use `python3`.

### DMInterpolationEvaluate_UW ImportError

If you see `undefined symbol: DMInterpolationEvaluate_UW`, the `_dminterp_wrapper` extension
is missing `petsc_tools.c`. This has been fixed in `setup.py`.

### JIT Compilation Errors (cc1 not found)

If you see `gcc: fatal error: cannot execute 'cc1': posix_spawnp: No such file or directory`:

The image is missing the `libexec` directory which contains compiler internal binaries.
Ensure the Dockerfile includes:
```dockerfile
COPY --from=builder --chown=jovyan:jovyan \
  /home/jovyan/underworld3/.pixi/envs/runtime/libexec \
  /home/jovyan/underworld3/.pixi/envs/runtime/libexec
```

The JIT compilation requires three directories:
- `include/` - Header files
- `x86_64-conda-linux-gnu/` - Compiler toolchain
- `libexec/` - Internal compiler binaries (cc1, etc.)

### VTK/Visualization Errors

The base image includes OSMesa for software rendering. Environment variables are set:
- `PYVISTA_OFF_SCREEN=true`
- `PYVISTA_USE_IPYVTK=true`
- `DISPLAY=:99`

## Technical Notes

### Why a Separate Launcher Repository?

mybinder.org caches container images by the **commit hash** of the repository. This means:
- Any commit to underworld3 → cache invalidated → slow rebuild
- Launcher repo rarely changes → cache stays valid → fast launches

The launcher repo acts as a stable "pointer" to the pre-built image, while underworld3
code is pulled fresh at runtime.

### Why Pixi?

Pixi provides:
- Fast, reproducible environment installation
- Conda-forge packages (PETSc, gmsh, etc.)
- Lock file for exact version pinning
- Cross-platform support

### Why Pre-built Image?

mybinder.org normally builds containers from scratch, which:
- Takes 10-20+ minutes for complex environments
- Can timeout on large dependencies
- Breaks caching when any file changes

Pre-built images:
- Launch in seconds (just pull + run)
- Only need rebuilding when dependencies change
- Code updates happen at runtime via start script

### Why Start Script Updates?

The start script pattern allows:
- Latest code without rebuilding the image
- Fast iteration during development
- Users always get current tutorials and fixes

Trade-off: ~30 second startup delay for `pixi run build`

## Limitations and Requirements

### mybinder.org Timeout

mybinder.org expects Jupyter to respond within **30 seconds** of container start. This constrains what can happen in the start script:

- ✅ `git pull` - fast (~2-3 seconds)
- ✅ `pixi run build` for Python-only changes - fast (~20-30 seconds)
- ❌ `pip install --force-reinstall` - too slow (rebuilds all Cython extensions, ~2-3 minutes)

### Cython Extension Changes Require Image Rebuild

**Critical**: Changes to Cython extensions (`.pyx` files) or their C dependencies (`petsc_tools.c`, etc.) require a **full image rebuild**. The runtime `pixi run build` cannot reliably rebuild these because:

1. Timestamp detection: pip sees pre-built `.so` files as newer than pulled source files
2. Timeout constraint: Forcing a full rebuild exceeds the 30-second mybinder timeout

**When to rebuild the base image:**
- Any changes to `setup.py` (extension definitions)
- Any changes to `.pyx` files in `src/underworld3/`
- Any changes to C files (`petsc_tools.c`, `petsc_tools.h`, etc.)
- Changes to `pixi.toml` dependencies

**What works with runtime pull:**
- Pure Python code changes (`.py` files)
- Documentation and notebook updates
- Test file changes

### Image Rebuild Procedure

When Cython or C code changes:

```bash
# Full rebuild (no cache) to ensure all extensions are recompiled
docker build --platform linux/amd64 --no-cache \
  -f container/Dockerfile.base.optimized \
  -t ghcr.io/underworldcode/uw3-base:2025.01-slim .

# Push to GHCR
docker push ghcr.io/underworldcode/uw3-base:2025.01-slim
```

The `--no-cache` flag is important to ensure fresh compilation of all Cython extensions.

### Start Script Output

The start script shows the last 3 lines of build output for debugging:
```bash
pixi run -e runtime build 2>&1 | tail -3 || true
```

If binder launches fail, check the Jupyter logs for build errors.

## Version History

| Tag | Date | Changes |
|-----|------|---------|
| 2025.01-slim | 2025-01-14 | Optimized image (~1.4GB vs 8.3GB): removed .git, headers, toolchain |
| 2025.01 | 2025-01-13 | Initial release with embedded start script, UW3_BRANCH support |

## Related Files

### In underworld3 repository
- `container/Dockerfile.base.optimized` - Optimized base image recipe (slim variant)
- `container/Dockerfile.base` - Full base image recipe (with headers/toolchain)
- `pixi.toml` - Pixi environment definition
- `pixi.lock` - Locked dependency versions
- `setup.py` - Cython extension build (includes petsc_tools.c fix)
- `docs/beginner/tutorials/` - Tutorial notebooks

### In uw3-binder-launcher repository
- `.binder/Dockerfile` - Minimal launcher config
- `README.md` - Launch badges and documentation
