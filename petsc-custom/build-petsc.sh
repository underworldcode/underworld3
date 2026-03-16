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
# MPI is auto-detected from the active pixi environment:
# - MPICH  → PETSC_ARCH = petsc-4-uw-mpich
# - OpenMPI → PETSC_ARCH = petsc-4-uw-openmpi
#
# Both builds co-exist under the same PETSc source tree.
# Build time: ~1 hour on Apple Silicon
#
# Usage:
#   ./build-petsc.sh          # Full build (clone, configure, build)
#   ./build-petsc.sh configure # Just reconfigure
#   ./build-petsc.sh build     # Just build (after configure)
#   ./build-petsc.sh petsc4py  # Just build petsc4py
#   ./build-petsc.sh clean     # Remove build for detected MPI
#   ./build-petsc.sh clean-all # Remove entire PETSc directory
#   ./build-petsc.sh help      # Show this help
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETSC_DIR="${SCRIPT_DIR}/petsc"

# Detect active pixi environment (robust)
if [ -z "$PIXI_PROJECT_ROOT" ]; then
    echo "Error: This script must be run from within a pixi environment"
    echo "Use: pixi run -e <env> ./build-petsc.sh"
    exit 1
fi

PIXI_ENV="$(python3 - <<'EOF'
import sys, pathlib
print(pathlib.Path(sys.executable).resolve().parents[1])
EOF
)"

# ── MPI auto-detection ──────────────────────────────────────────────
# Detect which MPI implementation is available in the pixi environment.
# Sets MPI_IMPL ("mpich" or "openmpi") and PETSC_ARCH accordingly.

detect_mpi() {
    local mpi_version
    mpi_version=$(python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())" 2>/dev/null || echo "")

    if echo "$mpi_version" | grep -qi "open mpi"; then
        echo "openmpi"
    elif echo "$mpi_version" | grep -qi "mpich"; then
        echo "mpich"
    else
        # Fallback: check for mpicc --version
        local mpicc_out
        mpicc_out=$("$PIXI_ENV/bin/mpicc" --version 2>&1 || echo "")
        if echo "$mpicc_out" | grep -qi "open mpi"; then
            echo "openmpi"
        else
            echo "mpich"  # default fallback
        fi
    fi
}

MPI_IMPL=$(detect_mpi)
PETSC_ARCH="petsc-4-uw-${MPI_IMPL}"

echo "=========================================="
echo "PETSc AMR Build Script"
echo "=========================================="
echo "PETSC_DIR:  $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo "MPI:        $MPI_IMPL"
echo "PIXI_ENV:   $PIXI_ENV"
echo "=========================================="

clone_petsc() {
    if [ -d "$PETSC_DIR" ]; then
        echo "PETSc directory already exists. Skipping clone."
        echo "To force fresh clone, run: ./build-petsc.sh clean-all"
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
    echo "Configuring PETSc with AMR tools ($MPI_IMPL)..."
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
        --download-ptscotch="${SCRIPT_DIR}/patches/scotch-7.0.10-c23-fix.tar.gz" \
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
        --download-openmpi=0 \
        --download-mpi4py=0 \
        --with-petsc4py=0

    echo "Configure complete."
}

build_petsc() {
    echo "Building PETSc ($MPI_IMPL)..."
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
    echo "Building petsc4py ($MPI_IMPL)..."
    cd "$PETSC_DIR/src/binding/petsc4py"

    export PETSC_DIR
    export PETSC_ARCH

    python setup.py build
    python setup.py install
    echo "petsc4py build complete."
}

clean_petsc() {
    # Clean just the arch-specific build
    local arch_dir="$PETSC_DIR/$PETSC_ARCH"
    echo "Removing PETSc build for $MPI_IMPL ($arch_dir)..."
    if [ -d "$arch_dir" ]; then
        rm -rf "$arch_dir"
        echo "Cleaned $PETSC_ARCH."
    else
        echo "Nothing to clean for $PETSC_ARCH."
    fi
}

clean_all() {
    echo "Removing entire PETSc directory..."
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
    echo "MPI auto-detected from pixi environment: $MPI_IMPL"
    echo "PETSC_ARCH: $PETSC_ARCH"
    echo ""
    echo "Commands:"
    echo "  (none)    Full build: clone, configure, build, petsc4py"
    echo "  clone     Clone PETSc repository"
    echo "  configure Configure PETSc with AMR tools"
    echo "  build     Build PETSc"
    echo "  test      Run PETSc tests"
    echo "  petsc4py  Build and install petsc4py"
    echo "  patch     Apply UW3 patches to PETSc source"
    echo "  clean     Remove build for current MPI ($PETSC_ARCH)"
    echo "  clean-all Remove entire PETSc directory (all MPI builds)"
    echo "  help      Show this help"
    echo ""
    echo "MPICH and OpenMPI builds co-exist. To build both:"
    echo "  pixi run -e amr         ./petsc-custom/build-petsc.sh"
    echo "  pixi run -e amr-openmpi ./petsc-custom/build-petsc.sh"
}

# Main entry point
case "${1:-all}" in
    all)
        clone_petsc
        apply_patches
        configure_petsc
        build_petsc
        build_petsc4py
        echo ""
        echo "=========================================="
        echo "PETSc AMR build complete! ($MPI_IMPL)"
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
    patch)
        apply_patches
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
    clean-all)
        clean_all
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
