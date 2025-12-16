#!/bin/bash
#
# Build PETSc with adaptive mesh refinement (AMR) tools
#
# This script builds a custom PETSc installation with:
# - pragmatic: anisotropic mesh adaptation
# - mmg: surface/volume mesh adaptation
# - parmmg: parallel mesh adaptation
# - slepc: eigenvalue solvers
# - mumps: direct solver
#
# Build time: ~1 hour on Apple Silicon
#
# Usage:
#   ./build-petsc.sh          # Full build (clone, configure, build)
#   ./build-petsc.sh configure # Just reconfigure
#   ./build-petsc.sh build     # Just build (after configure)
#   ./build-petsc.sh petsc4py  # Just build petsc4py
#   ./build-petsc.sh clean     # Remove PETSc directory
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETSC_DIR="${SCRIPT_DIR}/petsc"
PETSC_ARCH="petsc-4-uw"

# Detect pixi environment - need MPI and HDF5 from there
if [ -n "$PIXI_PROJECT_ROOT" ]; then
    PIXI_ENV="${PIXI_PROJECT_ROOT}/.pixi/envs/amr"
    if [ ! -d "$PIXI_ENV" ]; then
        PIXI_ENV="${PIXI_PROJECT_ROOT}/.pixi/envs/default"
    fi
else
    echo "Error: This script must be run from within a pixi environment"
    echo "Use: pixi run -e amr petsc-build"
    exit 1
fi

echo "=========================================="
echo "PETSc AMR Build Script"
echo "=========================================="
echo "PETSC_DIR:  $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo "PIXI_ENV:   $PIXI_ENV"
echo "=========================================="

clone_petsc() {
    if [ -d "$PETSC_DIR" ]; then
        echo "PETSc directory already exists. Skipping clone."
        echo "To force fresh clone, run: ./build-petsc.sh clean"
        return 0
    fi

    echo "Cloning PETSc release branch..."
    git clone -b release https://gitlab.com/petsc/petsc.git "$PETSC_DIR"
    echo "Clone complete."
}

configure_petsc() {
    echo "Configuring PETSc with AMR tools..."
    cd "$PETSC_DIR"

    # Configure with adaptive mesh refinement tools
    # Downloads: bison, eigen, metis, mmg, mumps, parmetis, parmmg,
    #            pragmatic, ptscotch, scalapack, slepc
    # Uses system: MPI (from pixi), HDF5 (from pixi)
    python3 ./configure \
        --with-petsc-arch="$PETSC_ARCH" \
        --download-bison \
        --download-eigen \
        --download-metis \
        --download-mmg \
        --download-mumps \
        --download-parmetis \
        --download-parmmg \
        --download-pragmatic \
        --download-ptscotch \
        --download-scalapack \
        --download-slepc \
        --with-debugging=0 \
        --with-hdf5=1 \
        --with-pragmatic=1 \
        --with-x=0 \
        --with-mpi-dir="$PIXI_ENV" \
        --with-hdf5-dir="$PIXI_ENV" \
        --download-hdf5=0 \
        --download-mpich=0 \
        --download-mpi4py=0 \
        --with-petsc4py=0

    echo "Configure complete."
}

build_petsc() {
    echo "Building PETSc..."
    cd "$PETSC_DIR"

    # Set environment for build
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

build_petsc4py() {
    echo "Building petsc4py..."
    cd "$PETSC_DIR/src/binding/petsc4py"

    export PETSC_DIR
    export PETSC_ARCH

    python setup.py build
    python setup.py install
    echo "petsc4py build complete."
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
    echo "  (none)    Full build: clone, configure, build, petsc4py"
    echo "  clone     Clone PETSc repository"
    echo "  configure Configure PETSc with AMR tools"
    echo "  build     Build PETSc"
    echo "  test      Run PETSc tests"
    echo "  petsc4py  Build and install petsc4py"
    echo "  clean     Remove PETSc directory"
    echo "  help      Show this help"
}

# Main entry point
case "${1:-all}" in
    all)
        clone_petsc
        configure_petsc
        build_petsc
        build_petsc4py
        echo ""
        echo "=========================================="
        echo "PETSc AMR build complete!"
        echo "Set these environment variables:"
        echo "  export PETSC_DIR=$PETSC_DIR"
        echo "  export PETSC_ARCH=$PETSC_ARCH"
        echo "=========================================="
        ;;
    clone)
        clone_petsc
        ;;
    configure)
        configure_petsc
        ;;
    build)
        build_petsc
        ;;
    test)
        test_petsc
        ;;
    petsc4py)
        build_petsc4py
        ;;
    clean)
        clean_petsc
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
