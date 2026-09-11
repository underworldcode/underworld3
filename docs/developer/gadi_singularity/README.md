# Building Underworld3 for Gadi (NCI)

This directory contains two Containerfiles to build the Underworld3 (UW3) Singularity image for Gadi (nci.org.au).

Both use Rocky Linux 8.10 to match Gadi's OS for ABI compatibility.

## Automated builds

Both images are built by GitHub Actions; the manual steps below are only needed for
local iteration or if you are building outside the `underworldcode` org.

| Workflow | Builds | Publishes | Runs when |
|---|---|---|---|
| `.github/workflows/gadi-uw3-image.yml` | `underworld3.rhel` | `ghcr.io/<owner>/underworld3-gadi:<branch\|tag>` (`:latest` on release) | push to `main`/`development` touching `src/**`, `pixi.toml`, `pyproject.toml`, `setup.py` or this directory; published releases; manual dispatch |
| `.github/workflows/gadi-petsc-image.yml` | `petsc.rhel` | `ghcr.io/<owner>/petsc-gadi:<version>-ompi` (+ `:latest`) | push touching `petsc.rhel` or `petsc-custom/patches/**`; manual dispatch |

PETSc is a ~25 min from-source build, so it is kept as a separate, rarely-triggered
workflow and the published image is reused as the base for every Underworld3 build.
To rebuild it explicitly:

```bash
gh workflow run gadi-petsc-image.yml -f petsc_version=3.25.0 -f make_np=4
```

To build the Underworld3 image from a specific ref:

```bash
gh workflow run gadi-uw3-image.yml -f uw3_branch=v3.1.0 -f image_tag=v3.1.0
```

Each build is followed by a smoke test: `import underworld3`, an import check for the
optional gmsh/vtk-osmesa stack that the Containerfile installs behind `|| echo skip`,
and a 2-rank MPI check.

### Retention

Only the **three most recent untagged** `underworld3-gadi` versions are kept (two for
`petsc-gadi`); older ones are pruned automatically after the smoke test passes. Tagged
images — `:latest`, `:main`, `:development`, and every `v*` release — are never pruned.

### Package visibility

A package published by these workflows inherits the repository's visibility, so on a
public repo both images are publicly pullable and Gadi compute nodes — which have no
GitHub credentials — can `singularity pull` them directly. A package created by hand
(`podman push` from a laptop) is **not** linked to the repository and does default to
private; set it public under its package settings, or authenticate the pull:

```bash
export SINGULARITY_DOCKER_USERNAME=<github-user>
export SINGULARITY_DOCKER_PASSWORD=<PAT with read:packages>
```

Note the workflows build single-platform `linux/amd64` images with buildx attestations
disabled (`provenance: false`, `sbom: false`). Attestations add an `unknown/unknown`
entry to the manifest index that some Singularity/Apptainer versions refuse to pull.

## Build Order

Build commands must be run from the top-level `underworld3/` directory (the build context).
Builds targeting Gadi must use `--platform linux/amd64`.

### 1. Build PETSc layer

```bash
podman build . \
    --platform linux/amd64 \
    --format docker \
    -t ghcr.io/<user>/petsc-gadi:3.25.0-ompi \
    -f ./docs/developer/gadi_singularity/petsc.rhel
```

### 2. Push PETSc image to registry

```bash
podman push ghcr.io/<user>/petsc-gadi:3.25.0-ompi
```

### 3. Build Underworld3

```bash
podman build . \
    --platform linux/amd64 \
    --format docker \
    --build-arg PETSC_IMAGE=ghcr.io/<user>/petsc-gadi:3.25.0-ompi \
    --build-arg UW3_BRANCH=development \
    -t ghcr.io/<user>/underworld3-gadi:latest \
    -f ./docs/developer/gadi_singularity/underworld3.rhel
```

### 4. Push Underworld3 image

```bash
podman push ghcr.io/<user>/underworld3-gadi:latest
```

## What Each File Does

- **petsc.rhel** — Builds PETSc 3.25.0 with full AMR support (petsc4py, slepc4py, mmg, parmmg, etc.)
- **underworld3.rhel** — Builds Underworld3 on top of the PETSc image

## Running on Gadi

Pull the image on Gadi (redirect cache to scratch to avoid home quota issues):

```bash
export SINGULARITY_CACHEDIR=/scratch/<project>/<user>/.singularity
module load singularity
singularity pull docker://ghcr.io/<user>/underworld3-gadi:latest
```

Run a script with MPI:

```bash
module load singularity
module load openmpi/4.1.7
mpiexec -n <ncpus> singularity exec underworld3-gadi_latest.sif python3 <script.py>
```

## Notes

- OpenFabrics (mlx5_0) warnings in the job error log are harmless
- PostHog telemetry failures on compute nodes are harmless (no outbound internet)
- The ghcr.io images must be set to **public** for Singularity to pull without authentication
