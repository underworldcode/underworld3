# Binder Container Setup for Underworld3

**Date**: 2025-01-14
**Status**: Implemented and working
**Current Image SHA**: `sha256:6d3894260f28dc837a21ce77a94bb1781f4ca12422934be7a4bd34b4a7b1223f`

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
| `2025.01-slim` | ~3.4GB | **Recommended** - Optimized for binder launches with JIT support |
| `2025.01` | ~8.3GB | Full image with all dev tools (for development) |

The slim image removes components not needed at runtime while **keeping JIT compilation support**:
- ✅ **Kept**: `include/` - Header files (required for JIT)
- ✅ **Kept**: `x86_64-conda-linux-gnu/` - Compiler toolchain (required for JIT)
- ✅ **Kept**: `libexec/` - Internal compiler binaries like cc1 (required for JIT)
- ❌ **Removed**: Git history (`.git` directory) - uses shallow clone instead
- ❌ **Removed**: Legacy documentation (`docs_legacy`)
- ❌ **Removed**: Package metadata (`conda-meta/`)
- ❌ **Removed**: Test directories in packages
- ❌ **Removed**: Man pages

## Critical: Layer Size Constraints for mybinder.org

**mybinder.org has an undocumented layer size limit of approximately 1GB**. Layers larger than ~1GB frequently fail to upload with timeouts or broken pipe errors.

### The Problem

The pixi runtime environment contains a ~2.7GB `lib` directory:
- `python3.12/site-packages/` - ~800MB
- `libLLVM*.so*`, `libclang*.so*` - ~355MB (LLVM libraries)
- `libvtk*.so*` - ~300MB (VTK libraries)
- `libgmsh*.so*`, `libopenblas*.so*`, `libQt*.so*`, `libicu*.so*` - ~400MB
- GCC, qt6, dri subdirectories - ~200MB
- Remaining shared libraries - ~700MB

A single `COPY --from=builder lib/ lib/` creates a 2.7GB layer that fails to push.

### The Solution: Split COPY Layers

The `Dockerfile.base.optimized` splits the lib directory into multiple layers:

**Builder stage - Organize files into split directories:**
```dockerfile
# Create split directories
RUN mkdir -p /home/jovyan/lib-split/llvm && \
    mkdir -p /home/jovyan/lib-split/vtk && \
    mkdir -p /home/jovyan/lib-split/other-large

# Move LLVM (~355MB)
RUN cd /home/jovyan/underworld3/.pixi/envs/runtime/lib && \
    mv libLLVM*.so* libclang*.so* /home/jovyan/lib-split/llvm/ 2>/dev/null || true

# Move VTK (~300MB)
RUN cd /home/jovyan/underworld3/.pixi/envs/runtime/lib && \
    mv libvtk*.so* libvisk*.so* /home/jovyan/lib-split/vtk/ 2>/dev/null || true

# Move other large libs (~400MB)
RUN cd /home/jovyan/underworld3/.pixi/envs/runtime/lib && \
    mv libgmsh*.so* libopenblas*.so* libopenvino*.so* \
       libQt*.so* libicu*.so* /home/jovyan/lib-split/other-large/ 2>/dev/null || true
```

**Final stage - Copy as separate layers:**
```dockerfile
# Layer 5a: Python + subdirs (~800MB)
COPY --from=builder .../lib/python3.12 .../lib/python3.12
COPY --from=builder .../lib/gcc .../lib/gcc
COPY --from=builder .../lib/qt6 .../lib/qt6
COPY --from=builder .../lib/dri .../lib/dri

# Layer 5b: VTK directories (~140MB)
COPY --from=builder .../lib/vtk-9.5 .../lib/vtk-9.5
COPY --from=builder .../lib/openvino-2025.2.0 .../lib/openvino-2025.2.0

# Layer 5c: LLVM libs (~350MB)
COPY --from=builder /home/jovyan/lib-split/llvm .../lib/

# Layer 5d: VTK libs (~300MB)
COPY --from=builder /home/jovyan/lib-split/vtk .../lib/

# Layer 5e: Other large libs (~400MB)
COPY --from=builder /home/jovyan/lib-split/other-large .../lib/

# Layer 5f: Remaining libs (~700MB)
COPY --from=builder .../lib .../lib
```

**Result**: All layers are under 800MB, which uploads reliably to GHCR and mybinder.org.

### Verifying Layer Sizes

After building, check layer sizes:
```bash
docker history ghcr.io/underworldcode/uw3-base:2025.01-slim --no-trunc
```

All layers should be under 1GB. If any layer exceeds ~800MB, consider splitting further.

## Launch URLs

### Quick Launch Badges

| Branch | Launch |
|--------|--------|
| `development` | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/development) |
| `main` | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/main) |

### URL Format

```
https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/<branch>
```

### Badge Markdown

```markdown
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/development)
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
export PATH="/home/jovyan/underworld3/.pixi/envs/runtime/bin:$PATH"
cd /home/jovyan/underworld3
exec "$@"
```

The script:
1. Sets up the pixi environment PATH directly (no pixi overhead at startup)
2. Changes to the underworld3 directory
3. Executes the passed command (typically jupyter lab)

**Note**: The current slim image uses a simplified start script that doesn't pull code at startup. The code is frozen at image build time. For development/testing branches that need fresh code, a more complex start script can be used:

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
- `LIBGL_ALWAYS_SOFTWARE=1`
- `MESA_GL_VERSION_OVERRIDE=3.3`
- `GALLIUM_DRIVER=llvmpipe`
- `VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow`

### PyVista 424 Errors on Binder (Trame Configuration)

If PyVista `pl.show()` displays "424 error - We can't seem to find the Binder page you are looking for":

**Problem**: Explicit `server_proxy_prefix` configuration interferes with mybinder.org's proxy setup.

**Solution**: Do NOT set explicit server proxy prefix. Let PyVista/trame auto-detect:

```python
# CORRECT - let PyVista auto-detect proxy settings
pv.global_theme.trame.server_proxy_enabled = True
# Don't set server_proxy_prefix - auto-detection works correctly

# WRONG - explicit prefix breaks on mybinder.org
pv.global_theme.trame.server_proxy_enabled = True
pv.global_theme.trame.server_proxy_prefix = "/proxy/"  # DON'T DO THIS
```

The Underworld3 `uw.visualisation.initialise()` function handles this correctly. If you see 424 errors, check that no code is setting `server_proxy_prefix` explicitly.

### Layer Upload Failures (Broken Pipe / Timeout)

If `docker push` fails with "write tcp: broken pipe" or hangs indefinitely:

**Problem**: One or more layers exceed mybinder.org's ~1GB limit.

**Solution**: Check layer sizes with `docker history` and split any layers over 800MB.

See [Critical: Layer Size Constraints](#critical-layer-size-constraints-for-mybinderorg) for details.

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

| Tag | Date | SHA | Changes |
|-----|------|-----|---------|
| 2025.01-slim | 2025-01-14 | `6d3894260f28...` | Split lib layers for reliable uploads, fixed trame proxy, kept JIT toolchain (~3.4GB) |
| 2025.01-slim | 2025-01-14 | (superseded) | Initial slim with JIT support but single large lib layer (upload failures) |
| 2025.01 | 2025-01-13 | — | Initial release with embedded start script, UW3_BRANCH support (~8.3GB) |

## Related Files

### In underworld3 repository
- `container/Dockerfile.base.optimized` - Optimized base image recipe (slim variant)
- `container/Dockerfile.base` - Full base image recipe (with headers/toolchain)
- `pixi.toml` - Pixi environment definition
- `pixi.lock` - Locked dependency versions
- `setup.py` - Cython extension build (includes petsc_tools.c fix)
- `docs/beginner/tutorials/` - Tutorial notebooks

### In uw3-binder-launcher repository
- `.binder/Dockerfile` - Minimal launcher config (must reference specific image SHA)
- `README.md` - Launch badges and documentation

## Generic Launcher for Any Repository

The launcher supports launching **any repository** with Underworld3 via nbgitpuller. Content repositories need NO special configuration.

### Using the Binder Wizard

Generate launch badges for your repository:

```bash
# Interactive wizard
python scripts/binder_wizard.py

# Quick generation
python scripts/binder_wizard.py username/my-course main tutorials/intro.ipynb
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Content Repository                                │
│                    (No .binder/ needed!)                             │
│                                                                      │
│   your-username/your-course                                         │
│   └── README.md  ← Just add a launch badge                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ Badge URL uses nbgitpuller
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    mybinder.org                                      │
│                                                                      │
│   1. Uses uw3-binder-launcher repo (cached UW3 image)                      │
│   2. nbgitpuller clones YOUR repo into workspace                    │
│   3. Opens JupyterLab with your notebooks                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### URL Format

```
https://mybinder.org/v2/gh/underworldcode/uw3-binder-launcher/development
  ?urlpath=git-pull
  ?repo=https://github.com/USER/REPO
  &branch=development
  &urlpath=lab/tree/REPO
```

### Benefits for Content Creators

| Traditional Binder | Generic Launcher |
|-------------------|------------------|
| Each repo needs .binder/ | No setup needed |
| 10-20 min build time | Instant launch |
| Cache per-repository | Single cached image |
| Maintain dependencies | Zero maintenance |

## Future: GitHub Actions Automation

To ensure the launcher repository stays synchronized with base image builds, consider implementing GitHub Actions automation:

### Proposed Workflow

```yaml
# .github/workflows/build-binder-image.yml (in underworld3 repo)
name: Build Binder Image

on:
  push:
    branches: [main, uw3-release-candidate]
    paths:
      - 'container/Dockerfile.base.optimized'
      - 'pixi.toml'
      - 'pixi.lock'
      - 'src/**/*.pyx'  # Cython changes require rebuild
      - 'setup.py'
  workflow_dispatch:  # Manual trigger

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: container/Dockerfile.base.optimized
          push: true
          platforms: linux/amd64
          tags: |
            ghcr.io/underworldcode/uw3-base:${{ github.ref_name }}
            ghcr.io/underworldcode/uw3-base:latest

      - name: Update launcher repository
        uses: peter-evans/repository-dispatch@v2
        with:
          token: ${{ secrets.LAUNCHER_PAT }}
          repository: underworldcode/uw3-binder-launcher
          event-type: image-updated
          client-payload: '{"branch": "${{ github.ref_name }}", "sha": "${{ steps.build.outputs.digest }}"}'
```

### Launcher Update Workflow

```yaml
# .github/workflows/update-image.yml (in uw3-binder-launcher repo)
name: Update Image Reference

on:
  repository_dispatch:
    types: [image-updated]

jobs:
  update-dockerfile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.client_payload.branch }}

      - name: Update Dockerfile
        run: |
          SHA="${{ github.event.client_payload.sha }}"
          # Update the SHA reference in Dockerfile
          sed -i "s/sha256:[a-f0-9]*/sha256:${SHA#sha256:}/" .binder/Dockerfile

      - name: Commit and push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git commit -am "Update image SHA to ${{ github.event.client_payload.sha }}"
          git push
```

### Requirements for Automation

1. **LAUNCHER_PAT**: Personal Access Token with `repo` scope for triggering launcher updates
2. **GHCR Permissions**: Package must be public or have appropriate access tokens
3. **Branch Protection**: Consider allowing bot commits to launcher repo

### Benefits of Automation

- **Consistency**: Launcher always references latest working image
- **Traceability**: Each launcher commit links to specific image SHA
- **Reliability**: Automated SHA updates prevent manual errors
- **CI Integration**: Can trigger binder tests after image updates
