# Building Underworld3 for Gadi (NCI)

This directory contains two Containerfiles to build the Underworld3 (UW3) Singularity image for Gadi (nci.org.au).

Both use Rocky Linux 8.10 to match Gadi's OS for ABI compatibility.

## Build Order

Build commands must be run from the top-level `underworld3/` directory (the build context).
Builds targeting Gadi must use `--platform linux/amd64`.

### 1. Build PETSc layer

```bash
podman build . \
    --platform linux/amd64 \
    --format docker \
    -t ghcr.io/<user>/petsc:3.25.0-ompi \
    -f ./docs/developer/gadi_singularity/petsc.rhel
```

### 2. Push PETSc image to registry

```bash
podman push ghcr.io/<user>/petsc:3.25.0-ompi
```

### 3. Build Underworld3

```bash
podman build . \
    --platform linux/amd64 \
    --format docker \
    --build-arg PETSC_IMAGE=ghcr.io/<user>/petsc:3.25.0-ompi \
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
