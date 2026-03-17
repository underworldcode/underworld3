#!/bin/bash
#
# Build PETSc with AMR tools for NCI Gadi (module OpenMPI + HDF5, PBS Pro)
#
# Differences from build-petsc.sh (local macOS/pixi):
#   MPI auto-detected from PATH (module load puts mpicc in PATH)
#   --with-hdf5-dir=$HDF5_DIR → uses Gadi's system HDF5 module (not pixi)
#   No --download-fblaslapack → Gadi has system BLAS/LAPACK
#   No --download-cmake       → cmake loaded from Gadi module
#   --with-petsc4py=1         → built during configure (not a separate step)
#
# This script builds the same AMR tool set as build-petsc.sh and build-petsc-kaiju.sh:
#   pragmatic, mmg, parmmg, slepc, mumps, metis, parmetis, ptscotch, scalapack
#
# Applies the same UW3 patches as build-petsc.sh:
#   plexfem-internal-boundary-ownership-fix.patch
#   scotch-7.0.10-c23-fix.tar.gz
#
# Usage (must be inside pixi gadi env with Gadi modules loaded):
#   module load openmpi/4.1.7 hdf5/1.12.2p cmake/3.31.6
#   source gadi_install_pixi.sh    (activates pixi gadi env)
#   ./build-petsc-gadi.sh          # Full build
#   ./build-petsc-gadi.sh configure # Just reconfigure
#   ./build-petsc-gadi.sh build     # Just make
#   ./build-petsc-gadi.sh patch     # Apply UW3 patches
#   ./build-petsc-gadi.sh test      # Run PETSc tests
#   ./build-petsc-gadi.sh clean     # Remove PETSc directory
#
# Build time: ~1 hour
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETSC_DIR="${SCRIPT_DIR}/petsc"
PETSC_ARCH="arch-linux-c-opt"

# Require Gadi OpenMPI to be loaded
if ! command -v mpicc &>/dev/null; then
    echo "Error: mpicc not found. Load Gadi OpenMPI module first:"
    echo "  module load openmpi/4.1.7"
    exit 1
fi

# Require HDF5_DIR to be set (from Gadi hdf5 module)
if [ -z "${HDF5_DIR}" ]; then
    echo "Error: HDF5_DIR is not set. Load Gadi HDF5 module first:"
    echo "  module load hdf5/1.12.2p"
    exit 1
fi

# Require pixi gadi environment
if ! echo "${PATH}" | tr ':' '\n' | grep -q "\.pixi/envs/gadi/bin"; then
    echo "Error: must be run inside the pixi gadi environment"
    echo "  source gadi_install_pixi.sh   (sets up env via pixi shell-hook)"
    exit 1
fi

echo "=========================================="
echo "PETSc AMR Build Script (Gadi)"
echo "=========================================="
echo "PETSC_DIR:  $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo "mpicc:      $(which mpicc)"
echo "HDF5_DIR:   $HDF5_DIR"
echo "=========================================="

clone_petsc() {
    if [ -d "$PETSC_DIR" ]; then
        echo "PETSc directory already exists. Skipping clone."
        echo "To force fresh clone, run: ./build-petsc-gadi.sh clean"
        return 0
    fi

    echo "Cloning PETSc release branch..."
    git clone -b release https://gitlab.com/petsc/petsc.git "$PETSC_DIR"
    echo "Clone complete."
}

apply_patches() {
    echo "Applying UW3 patches to PETSc..."
    cd "$PETSC_DIR"

    # Fix ghost facet ownership + part-consistent assembly in boundary
    # residual/integral/Jacobian paths (plexfem.c). Without this, internal
    # boundary natural BCs produce rank-dependent results in parallel.
    local patch="${SCRIPT_DIR}/patches/plexfem-internal-boundary-ownership-fix.patch"
    if [ -f "$patch" ]; then
        if git apply --check "$patch" 2>/dev/null; then
            git apply "$patch"
            echo "  Applied: plexfem-internal-boundary-ownership-fix.patch"
        else
            echo "  Skipped: plexfem-internal-boundary-ownership-fix.patch (already applied or conflict)"
        fi
    fi

    echo "Patches complete."
}

configure_petsc() {
    echo "Configuring PETSc with AMR tools..."
    cd "$PETSC_DIR"

    # Downloads and builds:
    #   AMR:        mmg, parmmg, pragmatic, eigen
    #   Solvers:    mumps, scalapack, slepc, superlu, superlu_dist, hypre
    #   Partitions: metis, parmetis, ptscotch (patched for C23)
    #   Mesh:       ctetgen, triangle, zlib
    #   HDF5:       from Gadi module (not downloaded)
    #   cmake:      from Gadi module (not downloaded)
    #   BLAS/LAPACK: from Gadi system (not downloaded)
    #   MPI:        from Gadi module (not downloaded)
    #   petsc4py:   built during configure
    python3 ./configure \
        --with-petsc-arch="${PETSC_ARCH}" \
        --with-debugging=0 \
        --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3" --FOPTFLAGS="-g -O3" \
        --with-shared-libraries=1 \
        --with-cxx-dialect=C++11 \
        --with-make-np=40 \
        --with-hdf5-dir="${HDF5_DIR}" \
        --with-hdf5=1 \
        --with-pragmatic=1 \
        --with-petsc4py=1 \
        --with-x=0 \
        --download-zlib=1 \
        --download-eigen=1 \
        --download-metis=1 \
        --download-parmetis=1 \
        --download-mumps=1 \
        --download-scalapack=1 \
        --download-slepc=1 \
        --download-ptscotch="${SCRIPT_DIR}/patches/scotch-7.0.10-c23-fix.tar.gz" \
        --download-mmg=1 \
        --download-mmg-cmake-arguments="-DMMG_INSTALL_PRIVATE_HEADERS=ON -DUSE_SCOTCH=OFF" \
        --download-parmmg=1 \
        --download-pragmatic=1 \
        --download-superlu=1 \
        --download-superlu_dist=1 \
        --download-hypre=1 \
        --download-ctetgen=1 \
        --download-triangle=1 \
        --useThreads=0

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
    echo "  (none)    Full build: clone, patch, configure, build"
    echo "  clone     Clone PETSc repository"
    echo "  patch     Apply UW3 patches to PETSc source"
    echo "  configure Configure PETSc with AMR tools"
    echo "  build     Build PETSc"
    echo "  test      Run PETSc tests"
    echo "  clean     Remove PETSc directory"
    echo "  help      Show this help"
}

case "${1:-all}" in
    all)
        clone_petsc
        apply_patches
        configure_petsc
        build_petsc
        echo ""
        echo "=========================================="
        echo "PETSc AMR build complete! (Gadi)"
        echo "Set these environment variables:"
        echo "  export PETSC_DIR=$PETSC_DIR"
        echo "  export PETSC_ARCH=$PETSC_ARCH"
        echo "  export PYTHONPATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$PYTHONPATH"
        echo "=========================================="
        ;;
    clone)     clone_petsc ;;
    patch)     apply_patches ;;
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
