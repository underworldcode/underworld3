# Binder Container Setup for Underworld3

**Date**: 2025-01-13
**Status**: Implemented and working

## Overview

Underworld3 uses a **pre-built container strategy** for mybinder.org launches. This approach provides fast startup times by avoiding the need for mybinder.org to build a new container on each launch.

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
│   ├── Start script (/home/jovyan/start)                            │
│   └── ENTRYPOINT configured                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ docker pull
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         mybinder.org                                │
│                                                                     │
│   1. Reads .binder/Dockerfile (single FROM line)                   │
│   2. Pulls pre-built image from GHCR                               │
│   3. Runs container with start script                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ on container start
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Start Script                                │
│                                                                     │
│   1. git pull origin uw3-release-candidate                         │
│   2. pixi run -e runtime build  (~30 seconds)                      │
│   3. Copy tutorials to workspace                                   │
│   4. exec pixi run -e runtime jupyter lab                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Benefits

1. **Fast Launches**: No Docker build required by mybinder.org - just pull and run
2. **Always Current**: Start script pulls latest code and rebuilds on each launch
3. **Cached Layers**: mybinder.org caches the pulled image at node level
4. **Simple Maintenance**: Update code in repo; users get latest on next launch

## File Locations

### In underworld3 repository

| File | Purpose |
|------|---------|
| `.binder/Dockerfile` | Minimal Dockerfile (single FROM line) for mybinder.org |
| `container/Dockerfile.base` | Recipe for building the base image |

### On GHCR

| Image | Description |
|-------|-------------|
| `ghcr.io/underworldcode/uw3-base:2025.01` | Pre-built base image with full environment |

## .binder/Dockerfile

The binder Dockerfile is intentionally minimal:

```dockerfile
# Binder Dockerfile - uses pre-built base image from GHCR
# The base image contains everything: pixi, underworld3, start script, ENTRYPOINT
# No additional build steps needed - just pull and run
FROM ghcr.io/underworldcode/uw3-base:2025.01
```

This single `FROM` line means mybinder.org doesn't need to build any additional layers.

## Base Image Contents

The base image (`container/Dockerfile.base`) includes:

### System Dependencies
- Ubuntu 24.04
- OpenGL/Mesa libraries for visualization
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
- Cloned from `uw3-release-candidate` branch
- Built and installed in pixi environment
- Jupyter kernel registered as "Underworld3"

### Start Script
Embedded at `/home/jovyan/start`:

```bash
#!/bin/bash
cd /home/jovyan/underworld3
git pull origin uw3-release-candidate 2>/dev/null || true
pixi run -e runtime build 2>/dev/null || true
cp -r docs/beginner/tutorials/* /home/jovyan/Tutorials/ 2>/dev/null || true
exec pixi run -e runtime "$@"
```

### Binder Configuration
- User: `jovyan` (UID 1000) - mybinder.org standard
- ENTRYPOINT: `/home/jovyan/start`
- CMD: `jupyter lab --ip=0.0.0.0 --no-browser`
- Tutorials pre-copied to `/home/jovyan/Tutorials/`

## Launch URL

```
https://mybinder.org/v2/gh/underworldcode/underworld3/uw3-release-candidate
```

### Badge for Documentation

```markdown
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/underworld3/uw3-release-candidate)
```

## Maintenance

### Updating Code (No Image Rebuild Needed)

For routine code changes:
1. Push changes to `uw3-release-candidate` branch
2. Users launching binder will automatically get latest code
3. The start script runs `git pull` and rebuilds on each launch

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

5. **Update .binder/Dockerfile** with new tag:
   ```dockerfile
   FROM ghcr.io/underworldcode/uw3-base:YYYY.MM
   ```

6. **Commit and push** the updated Dockerfile

### GHCR Permissions

The GHCR package must be **public** for mybinder.org to pull it without authentication.

To make public:
1. Go to https://github.com/orgs/underworldcode/packages/container/uw3-base/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Public"

## Troubleshooting

### "401 Unauthorized" on mybinder.org

The GHCR image is private. Make it public (see above).

### "start: not found" error

The COPY path in Dockerfile is wrong. Ensure paths are relative to repo root, not `.binder/` directory.

### Platform mismatch (linux-aarch64)

Building on Apple Silicon defaults to ARM. Use `--platform linux/amd64`:
```bash
docker build --platform linux/amd64 -f container/Dockerfile.base ...
```

### Slow launches

If mybinder.org is rebuilding the image, check that:
1. `.binder/Dockerfile` is minimal (single FROM line)
2. No other config files in `.binder/` that might trigger a build
3. The base image tag matches exactly

## Technical Notes

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
| 2025.01 | 2025-01-13 | Initial release with embedded start script |

## Related Files

- `pixi.toml` - Pixi environment definition
- `pixi.lock` - Locked dependency versions
- `docs/beginner/tutorials/` - Tutorial notebooks copied to workspace
