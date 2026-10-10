# HPC containers for Underworld3

Singularity/Apptainer images for the HPC systems we run on, built by GitHub Actions and
published to GHCR. Site-specific run instructions (job templates, MPI settings, shared
image locations) live in
[underworld-community/uw3-hpc-install-run](https://github.com/underworld-community/uw3-hpc-install-run);
this directory is the recipes.

## Layout

One platform base per OS/MPI family; everything above it is shared.

```
base.rocky-ompi    Rocky 8.10 + system OpenMPI      -> uw3-base-rocky-ompi:{runtime,builder}     Gadi, Kaiju
base.ubuntu-mpich  Ubuntu 24.04 + MPICH 3.4.3       -> uw3-base-ubuntu-mpich:{runtime,builder}   Setonix
petsc.hpc          PETSc from source, on a base     -> petsc-{gadi,setonix}:<version>-{ompi,mpich}
underworld3.hpc    UW3 + Python stack, on PETSc     -> underworld3-{gadi,setonix}:<branch|tag>
```

`petsc.hpc` and `underworld3.hpc` never call a package manager; they compile against
whatever `mpicc` the `:builder` base provides and lay the result over the `:runtime` base.
Adding a platform means adding a `base.*` file and a matrix row in each workflow.

| Platform | Base | MPI in image | On the machine |
|---|---|---|---|
| Gadi | Rocky 8.10 | OpenMPI 4.1.1 (`libmpi.so.40`) | host `openmpi/4.1.7` injected via `LD_LIBRARY_PATH` — measured equal to bare metal |
| Kaiju | same image as Gadi | OpenMPI 4.1.1 | runs as-is under `srun --mpi=pmix` |
| Setonix | Ubuntu 24.04 | MPICH 3.4.3 (`libmpi.so.12`) | host Cray MPICH bind-mounted by `singularity/4.1.0-mpi` |

Setonix needs 3.4.3 specifically: Cray MPICH 8.1 is MPICH-3.4-based, and code compiled
against MPICH 4.x headers references MPI-4 symbols the host library lacks.

## Automated builds

| Workflow | Builds | Runs when |
|---|---|---|
| `hpc-base-images.yml` | `base.*` | push touching `base.*`; dispatch |
| `hpc-petsc-image.yml` | `petsc.hpc` per platform | push touching `petsc.hpc` or `petsc-custom/patches/**`; dispatch |
| `hpc-uw3-image.yml` | `underworld3.hpc` per platform | push to `main`/`development` touching `src/**`, `pixi.toml`, `pyproject.toml`, `setup.py` or `underworld3.hpc`; `v*` tags; releases; dispatch |

PETSc is ~25 min per platform, so it is a separate, rarely-triggered workflow and its
image is reused by every UW3 build. After a base change, rebuild PETSc, then UW3:

```bash
gh workflow run hpc-petsc-image.yml -f petsc_version=3.25.0 -f make_np=4
gh workflow run hpc-uw3-image.yml -f uw3_branch=v3.1.0 -f image_tag=v3.1.0
```

Every build is smoke-tested before anything is pruned: `import underworld3`, the optional
gmsh/vtk-osmesa stack, and a 2-rank MPI run that asserts the MPI flavour (Open MPI vs
MPICH) — a silently wrong `libmpi` is the failure that would otherwise only show on a
compute node.

Retention: the three most recent untagged `underworld3-*` versions are kept (two for
`petsc-*` and the bases). Tagged images are never pruned.

Images inherit the repository's visibility. Packages created by hand (`podman push`) are
not repo-linked and default to private; make them public or the compute nodes cannot pull.

The workflows publish single-platform `linux/amd64` with attestations off
(`provenance: false`, `sbom: false`); attestations add an `unknown/unknown` manifest entry
that some Singularity versions refuse.

## Building by hand

From the repository root, x86_64 only:

```bash
podman build . --platform linux/amd64 --format docker --target builder \
    -t ghcr.io/<user>/uw3-base-rocky-ompi:builder -f docs/developer/hpc_containers/base.rocky-ompi
podman build . --platform linux/amd64 --format docker --target runtime \
    -t ghcr.io/<user>/uw3-base-rocky-ompi:runtime -f docs/developer/hpc_containers/base.rocky-ompi

podman build . --platform linux/amd64 --format docker \
    --build-arg BASE_BUILDER=ghcr.io/<user>/uw3-base-rocky-ompi:builder \
    --build-arg BASE_RUNTIME=ghcr.io/<user>/uw3-base-rocky-ompi:runtime \
    -t ghcr.io/<user>/petsc-gadi:3.25.0-ompi -f docs/developer/hpc_containers/petsc.hpc

podman build . --platform linux/amd64 --format docker \
    --build-arg PETSC_IMAGE=ghcr.io/<user>/petsc-gadi:3.25.0-ompi \
    --build-arg BASE_BUILDER=ghcr.io/<user>/uw3-base-rocky-ompi:builder \
    --build-arg UW3_BRANCH=development \
    -t ghcr.io/<user>/underworld3-gadi:development -f docs/developer/hpc_containers/underworld3.hpc
```

Substitute `ubuntu-mpich` / `setonix` / `mpich` for the Setonix chain. Pass
`--build-arg PETSC_MAKE_NP=2` inside a memory-constrained podman machine.

## Pulling

```bash
export SINGULARITY_CACHEDIR=/scratch/<project>/<user>/.singularity   # not $HOME
module load singularity                                              # singularity/4.1.0-mpi on Setonix
singularity pull docker://ghcr.io/underworldcode/underworld3-gadi:latest
```

How to run on each system is in the `uw3-hpc-install-run` repo.
