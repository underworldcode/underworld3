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
FROM ghcr.io/underworldcode/uw3-base:2025.01

# Pull from <branch> branch of underworld3
ENV UW3_BRANCH=<branch>
```

The `UW3_BRANCH` environment variable tells the start script which branch to pull.

## Base Image Contents

The base image (`container/Dockerfile.base`) includes:

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

### Updating Dependencies (Image Rebuild Required)

When dependencies change (new packages, version updates):

1. **Update pixi.toml** with new dependencies

2. **Rebuild base image**:
   ```bash
   docker build --platform linux/amd64 \
     -f container/Dockerfile.base \
     -t ghcr.io/underworldcode/uw3-base:YYYY.MM .
   ```

3. **Test locally**:
   ```bash
   docker run --rm -p 8888:8888 ghcr.io/underworldcode/uw3-base:YYYY.MM
   ```

4. **Push to GHCR**:
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
   docker push ghcr.io/underworldcode/uw3-base:YYYY.MM
   ```

5. **Update all launcher branches** with new tag:
   ```bash
   cd uw3-binder-launcher
   for branch in main uw3-release-candidate development; do
     git checkout $branch
     # Edit .binder/Dockerfile with new tag
     git commit -am "Update to uw3-base:YYYY.MM"
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
1. First launch after cache expiry will pull the ~6GB image (takes a few minutes)
2. Subsequent launches should be faster (image cached at node level)
3. The ~30 second `pixi run build` happens every launch

### Kernel not found

Notebooks must use the `python3` kernel name, not `pixi-kernel-python3`.
Tutorial notebooks have been normalized to use `python3`.

### DMInterpolationEvaluate_UW ImportError

If you see `undefined symbol: DMInterpolationEvaluate_UW`, the `_dminterp_wrapper` extension
is missing `petsc_tools.c`. This has been fixed in `setup.py`.

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

## Version History

| Tag | Date | Changes |
|-----|------|---------|
| 2025.01 | 2025-01-13 | Initial release with embedded start script, UW3_BRANCH support |

## Related Files

### In underworld3 repository
- `container/Dockerfile.base` - Base image recipe
- `pixi.toml` - Pixi environment definition
- `pixi.lock` - Locked dependency versions
- `setup.py` - Cython extension build (includes petsc_tools.c fix)
- `docs/beginner/tutorials/` - Tutorial notebooks

### In uw3-binder-launcher repository
- `.binder/Dockerfile` - Minimal launcher config
- `README.md` - Launch badges and documentation
