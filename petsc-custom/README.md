# PETSc Custom Build for AMR

This directory contains the build infrastructure for a custom PETSc installation with adaptive mesh refinement (AMR) tools.

## Why Custom PETSc?

The conda-forge PETSc package is great for most users but lacks:
- **pragmatic**: Anisotropic mesh adaptation
- **mmg**: Surface and volume mesh adaptation
- **parmmg**: Parallel mesh adaptation

These tools are essential for adaptive mesh refinement research workflows.

## Build Time

Expect ~1 hour on Apple Silicon M1/M2/M3.

## Usage

```bash
# Build PETSc with AMR tools (from repository root)
pixi run -e amr petsc-build

# Or run the script directly (must be in pixi shell)
pixi shell -e amr
cd petsc-custom
./build-petsc.sh
```

## Build Commands

```bash
./build-petsc.sh           # Full build (clone, configure, build, petsc4py)
./build-petsc.sh clone     # Just clone PETSc
./build-petsc.sh configure # Reconfigure
./build-petsc.sh build     # Just make
./build-petsc.sh petsc4py  # Just build petsc4py
./build-petsc.sh test      # Run PETSc tests
./build-petsc.sh clean     # Remove everything
```

## What Gets Installed

The build downloads and compiles:
- **bison**: Parser generator (build dependency)
- **eigen**: Template library for linear algebra
- **metis/parmetis**: Graph partitioning
- **mmg/parmmg**: Mesh adaptation
- **mumps**: Direct sparse solver
- **pragmatic**: Anisotropic mesh adaptation
- **ptscotch**: Parallel graph partitioning
- **scalapack**: Parallel linear algebra
- **slepc**: Eigenvalue problem solvers

## Directory Structure After Build

```
petsc-custom/
├── build-petsc.sh      # Build script
├── README.md           # This file
└── petsc/              # PETSc source and build (created by build)
    ├── petsc-4-uw/     # Build output (PETSC_ARCH)
    └── src/
        └── binding/
            └── petsc4py/
```

## Important Notes

1. **Immovable**: Once built, PETSc cannot be relocated. Paths are hardcoded.
2. **Disk Space**: Requires ~5GB for full build.
3. **Rebuild Required**: If you move the repository, you must rebuild.

## Environment Variables

The AMR environment automatically sets:
```bash
PETSC_DIR=$PIXI_PROJECT_ROOT/petsc-custom/petsc
PETSC_ARCH=petsc-4-uw
```
