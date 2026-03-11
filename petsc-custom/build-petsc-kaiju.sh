#!/bin/bash
#
# Build PETSc with AMR tools for the Kaiju cluster (Rocky Linux 8, Spack OpenMPI)
#
# Differences from build-petsc.sh (local macOS/pixi):
#   MPI auto-detected from PATH (spack load puts mpicc in PATH; no --with-mpi-dir needed)
#   --download-hdf5  → PETSc downloads HDF5 (not provided by pixi)
#   --download-fblaslapack → no guaranteed system BLAS on Rocky Linux 8
#   --download-cmake → spack does not have cmake
#   --with-petsc4py  → built during configure (not a separate step)
#
# This script builds the same AMR tool set as build-petsc.sh:
#   pragmatic, mmg, parmmg, slepc, mumps, metis, parmetis, ptscotch, scalapack
#
# Usage (must be inside a pixi kaiju shell with spack OpenMPI loaded):
#   spack load openmpi@4.1.6
#   pixi shell -e kaiju
#   ./build-petsc-kaiju.sh            # Full build
#   ./build-petsc-kaiju.sh configure  # Just reconfigure
#   ./build-petsc-kaiju.sh build      # Just make
#   ./build-petsc-kaiju.sh clean      # Remove PETSc directory
#
# Build time: ~1 hour
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETSC_DIR="${SCRIPT_DIR}/petsc"
PETSC_ARCH="petsc-4-uw"

# Require spack OpenMPI to be loaded
if ! command -v mpicc &>/dev/null; then
    echo "Error: mpicc not found. Load spack OpenMPI first:"
    echo "  spack load openmpi@4.1.6"
    exit 1
fi

# Require pixi kaiju environment
# Check PATH since PIXI_ENVIRONMENT is not set by pixi shell-hook (only by pixi shell)
if ! echo "$PATH" | tr ':' '\n' | grep -q "\.pixi/envs/kaiju/bin"; then
    echo "Error: must be run inside the pixi kaiju environment"
    echo "  source uw3_install_kaiju_amr.sh   (sets up env via pixi shell-hook)"
    exit 1
fi

echo "=========================================="
echo "PETSc AMR Build Script (Kaiju)"
echo "=========================================="
echo "PETSC_DIR:  $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo "mpicc:      $(which mpicc)"
echo "=========================================="

clone_petsc() {
    if [ -d "$PETSC_DIR" ]; then
        echo "PETSc directory already exists. Skipping clone."
        echo "To force fresh clone, run: ./build-petsc-kaiju.sh clean"
        return 0
    fi

    echo "Cloning PETSc release branch..."
    git clone -b release https://gitlab.com/petsc/petsc.git "$PETSC_DIR"
    echo "Clone complete."
}

configure_petsc() {
    echo "Configuring PETSc with AMR tools..."
    cd "$PETSC_DIR"

    # Downloads and builds:
    #   AMR:        mmg, parmmg, pragmatic, eigen, bison
    #   Solvers:    mumps, scalapack, slepc
    #   Partitions: metis, parmetis, ptscotch
    #   BLAS/LAPACK: fblaslapack (Rocky Linux 8 has no guaranteed system BLAS)
    #   HDF5:       downloaded (not provided by pixi in kaiju env)
    #   cmake:      downloaded (spack does not have cmake)
    #   MPI:        spack OpenMPI (not downloaded)
    #   petsc4py:   built during configure
    # MPI_DIR is computed from `which mpicc` (spack OpenMPI in PATH).
    # LD_LIBRARY_PATH must include $MPI_DIR/lib so PETSc configure test binaries
    # can find libmpi.so at runtime (spack uses RPATH for its own binaries but
    # does not set LD_LIBRARY_PATH — load_env in uw3_install_kaiju_amr.sh sets it).
    MPI_DIR="$(dirname "$(dirname "$(which mpicc)")")"
    python3 ./configure \
        --with-petsc-arch="$PETSC_ARCH" \
        --with-debugging=0 \
        --with-mpi-dir="$MPI_DIR" \
        --download-hdf5=1 \
        --download-fblaslapack=1 \
        --download-cmake=1 \
        --download-bison=1 \
        --download-eigen=1 \
        --download-metis=1 \
        --download-parmetis=1 \
        --download-mumps=1 \
        --download-scalapack=1 \
        --download-slepc=1 \
        --download-ptscotch=1 \
        --download-mmg=1 \
        --download-parmmg=1 \
        --download-pragmatic=1 \
        --with-pragmatic=1 \
        --with-petsc4py=1 \
        --with-x=0 \
        --with-make-np=40

    echo "Configure complete."
}

build_petsc() {
    echo "Building PETSc..."
    cd "$PETSC_DIR"

    export PETSC_DIR
    export PETSC_ARCH

    make all
    echo "PETSc build complete."
}

test_petsc() {
    echo "Testing PETSc..."
    cd "$PETSC_DIR"

    export PETSC_DIR
    export PETSC_ARCH

    make check
    echo "PETSc tests complete."
}

clean_petsc() {
    echo "Removing PETSc directory..."
    if [ -d "$PETSC_DIR" ]; then
        rm -rf "$PETSC_DIR"
        echo "Cleaned."
    else
        echo "Nothing to clean."
    fi
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  (none)    Full build: clone, configure, build"
    echo "  clone     Clone PETSc repository"
    echo "  configure Configure PETSc with AMR tools"
    echo "  build     Build PETSc"
    echo "  test      Run PETSc tests"
    echo "  clean     Remove PETSc directory"
    echo "  help      Show this help"
}

case "${1:-all}" in
    all)
        clone_petsc
        configure_petsc
        build_petsc
        echo ""
        echo "=========================================="
        echo "PETSc AMR build complete!"
        echo "Set these environment variables:"
        echo "  export PETSC_DIR=$PETSC_DIR"
        echo "  export PETSC_ARCH=$PETSC_ARCH"
        echo "  export PYTHONPATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$PYTHONPATH"
        echo "=========================================="
        ;;
    clone)     clone_petsc ;;
    configure) configure_petsc ;;
    build)     build_petsc ;;
    test)      test_petsc ;;
    clean)     clean_petsc ;;
    help|--help|-h) show_help ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
