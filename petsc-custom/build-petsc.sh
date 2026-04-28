#!/bin/bash
#
# Build PETSc with adaptive mesh refinement (AMR) tools
#
# Supports three build targets, auto-detected from hostname or UW_CLUSTER env var:
#   local  — macOS/Linux developer machine (pixi env for MPI and HDF5)
#   kaiju  — Kaiju cluster (Rocky Linux 8, Spack OpenMPI, no system HDF5/cmake/BLAS)
#   gadi   — NCI Gadi (CentOS, module OpenMPI + HDF5, PBS Pro)
#
# Cluster-specific differences:
#
#   Aspect        local                    kaiju             gadi
#   PETSC_ARCH    petsc-4-uw-{mpich,       petsc-4-uw-       petsc-4-uw-
#                   openmpi}               openmpi           openmpi
#   MPI           pixi env                 spack (PATH)      module (PATH)
#   HDF5          pixi env                 download          module ($HDF5_DIR)
#   BLAS/LAPACK   auto                     download          download (auto fails)
#   cmake         pixi env                 download          module
#   bison         download                 download          system
#   petsc4py      separate step            with-petsc4py=1   with-petsc4py=1
#   extra flags   —                        —                 superlu, hypre, ...
#
# Override auto-detection: export UW_CLUSTER=local|kaiju|gadi
#
# Usage:
#   ./build-petsc.sh           # Full build (clone, patch, configure, build)
#   ./build-petsc.sh configure # Just reconfigure
#   ./build-petsc.sh build     # Just build (after configure)
#   ./build-petsc.sh petsc4py  # Build petsc4py separately (local only)
#   ./build-petsc.sh patch     # Apply UW3 patches
#   ./build-petsc.sh test      # Run PETSc tests
#   ./build-petsc.sh clean     # Remove build for current arch
#   ./build-petsc.sh clean-all # Remove entire PETSc directory
#   ./build-petsc.sh help      # Show this help
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PETSC_DIR="${SCRIPT_DIR}/petsc"
VERSION_FILE="${SCRIPT_DIR}/.petsc-version"

# ── PETSc version detection ──────────────────────────────────────────────────
# Read version from .petsc-version file, or detect from the git checkout.
# Returns short form like "324" for v3.24.x or "325" for v3.25.x.

petsc_version_short() {
    local ver=""
    if [ -f "${VERSION_FILE}" ]; then
        ver=$(cat "${VERSION_FILE}" | tr -d '[:space:]')
    fi
    if [ -z "${ver}" ] && [ -d "${PETSC_DIR}/.git" ]; then
        # Detect from git tag: v3.24.2 → 324
        local tag
        tag=$(cd "${PETSC_DIR}" && git describe --tags --abbrev=0 2>/dev/null || echo "")
        if [ -n "${tag}" ]; then
            ver=$(echo "${tag}" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\1\2/')
        fi
    fi
    echo "${ver:-324}"  # default to 324 if detection fails
}

PETSC_VER=$(petsc_version_short)

# ── Cluster detection ─────────────────────────────────────────────────────────
detect_cluster() {
    if [ -n "${UW_CLUSTER}" ]; then
        echo "${UW_CLUSTER}"; return
    fi
    local hn
    hn="$(hostname -f 2>/dev/null || hostname)"
    case "${hn}" in
        *.gadi.nci.org.au|gadi-*) echo "gadi" ;;
        kaiju*)                    echo "kaiju" ;;
        *)                         echo "local" ;;
    esac
}

CLUSTER="$(detect_cluster)"

# ── Cluster-specific configuration ───────────────────────────────────────────
# Sets PETSC_ARCH, MPI_IMPL, and cluster-specific variables (PIXI_ENV, MPI_DIR,
# HDF5_DIR). Also validates that the required environment is active.

case "${CLUSTER}" in
    local)
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

        _detect_local_mpi() {
            local mpi_version
            mpi_version=$(python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())" 2>/dev/null || echo "")
            if echo "$mpi_version" | grep -qi "open mpi"; then
                echo "openmpi"
            elif echo "$mpi_version" | grep -qi "mpich"; then
                echo "mpich"
            else
                local mpicc_out
                mpicc_out=$("$PIXI_ENV/bin/mpicc" --version 2>&1 || echo "")
                if echo "$mpicc_out" | grep -qi "open mpi"; then
                    echo "openmpi"
                else
                    echo "mpich"
                fi
            fi
        }

        MPI_IMPL=$(_detect_local_mpi)
        PETSC_ARCH="petsc-${PETSC_VER}-uw-${MPI_IMPL}"
        ;;

    kaiju)
        if ! command -v mpicc &>/dev/null; then
            echo "Error: mpicc not found. Load spack OpenMPI first:"
            echo "  spack load openmpi@4.1.6"
            exit 1
        fi
        if ! echo "${PATH}" | tr ':' '\n' | grep -q "\.pixi/envs/hpc/bin"; then
            echo "Error: must be run inside the pixi hpc environment"
            echo "  source kaiju_install_user.sh   (activates env via pixi shell-hook)"
            exit 1
        fi
        MPI_DIR="$(dirname "$(dirname "$(which mpicc)")")"
        MPI_IMPL="openmpi"
        PETSC_ARCH="petsc-${PETSC_VER}-uw-openmpi"
        ;;

    gadi)
        if ! command -v mpicc &>/dev/null; then
            echo "Error: mpicc not found. Load Gadi OpenMPI module first:"
            echo "  module load openmpi/4.1.7"
            exit 1
        fi
        if [ -z "${HDF5_DIR}" ]; then
            echo "Error: HDF5_DIR is not set. Load Gadi HDF5 module first:"
            echo "  module load hdf5/1.12.2p"
            exit 1
        fi
        if ! echo "${PATH}" | tr ':' '\n' | grep -q "\.pixi/envs/hpc/bin"; then
            echo "Error: must be run inside the pixi hpc environment"
            echo "  source gadi_install_shared.sh   (activates env via pixi shell-hook)"
            exit 1
        fi
        MPI_DIR="$(dirname "$(dirname "$(which mpicc)")")"
        MPI_IMPL="openmpi"
        PETSC_ARCH="petsc-${PETSC_VER}-uw-openmpi"
        ;;

    *)
        echo "Unknown cluster: ${CLUSTER}"
        echo "Set UW_CLUSTER=local|kaiju|gadi to override auto-detection"
        exit 1
        ;;
esac

echo "=========================================="
echo "PETSc AMR Build Script"
echo "=========================================="
echo "CLUSTER:    ${CLUSTER}"
echo "PETSC_DIR:  ${PETSC_DIR}"
echo "PETSC_ARCH: ${PETSC_ARCH}"
echo "MPI:        ${MPI_IMPL}"
if [ "${CLUSTER}" = "local" ]; then
    echo "PIXI_ENV:   ${PIXI_ENV}"
else
    echo "MPI_DIR:    ${MPI_DIR}"
fi
[ "${CLUSTER}" = "gadi" ] && echo "HDF5_DIR:   ${HDF5_DIR}"
echo "=========================================="

# ── Gadi-specific: build environment setup ───────────────────────────────────
# Handles compiler-tagged Fortran MPI libs and conda toolchain interference.
# Must be called before any compile/link step on Gadi.
setup_gadi_build_env() {
    if [ -z "${MPI_DIR}" ]; then
        echo "Error: MPI_DIR is not set. Source gadi_install_shared.sh first."
        exit 1
    fi

    # Create symlinks for Gadi's compiler-tagged Fortran MPI libs.
    # mpifort --showme refers to libmpi_usempif08 etc. (no compiler tag),
    # but Gadi only ships _GNU, _Intel, _nvidia variants.
    local _mpi_gnu_dir="${SCRIPT_DIR}/mpi-gadi-gnu-libs"
    mkdir -p "${_mpi_gnu_dir}"
    for _lib in usempif08 usempi_ignore_tkr mpifh; do
        [ ! -f "${_mpi_gnu_dir}/libmpi_${_lib}.so" ] && \
            ln -sf "${MPI_DIR}/lib/libmpi_${_lib}_GNU.so" "${_mpi_gnu_dir}/libmpi_${_lib}.so"
    done

    # LD_LIBRARY_PATH = runtime search path (dynamic loader)
    # LIBRARY_PATH    = link-time search path (ld resolves -lmpi_usempif08 etc.)
    export LD_LIBRARY_PATH="${_mpi_gnu_dir}:${MPI_DIR}/lib:/apps/ucc/1.3.0/lib:/usr/lib64:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${_mpi_gnu_dir}:${LIBRARY_PATH}"

    # Unset conda/pixi compiler vars that interfere with OpenMPI wrappers.
    # The pixi hpc env ships a full conda toolchain (x86_64-conda-linux-gnu-*)
    # that conflicts with system compilers required by Gadi's OpenMPI.
    unset CC CXX FC F77 F90 CPP AR RANLIB
    unset CFLAGS CXXFLAGS FFLAGS CPPFLAGS LDFLAGS

    # Force MPI wrappers to use system compilers, not conda's gcc
    export OMPI_CC=/usr/bin/gcc
    export OMPI_CXX=/usr/bin/g++
    export OMPI_FC=/usr/bin/gfortran
    # Gadi puts Fortran MPI headers in a compiler-tagged subdirectory (include/GNU/)
    export OMPI_FCFLAGS="-I${MPI_DIR}/include/GNU"

    # Put system bin dirs first so the system linker (/usr/bin/ld) wins over
    # conda's ld — conda's ld cannot find Gadi-specific libs (hcoll, ucc, libnl).
    export PATH="/usr/bin:/usr/local/bin:${MPI_DIR}/bin:${PATH}"
}

# ── Local macOS/OpenMPI build environment setup ──────────────────────────────
# The pixi osx-arm64 toolchain currently mixes a conda clang/ld stack, OpenMPI
# wrapper defaults, and Fortran settings that are not coherent enough for PETSc's
# mixed C/Fortran configure checks. For local macOS OpenMPI builds, force the
# MPI wrappers onto Apple's clang/clang++ and a working external gfortran.
setup_local_macos_openmpi_build_env() {
    [ "${CLUSTER}" = "local" ] || return 0
    [ "${MPI_IMPL}" = "openmpi" ] || return 0
    [ "$(uname -s)" = "Darwin" ] || return 0

    local _sdkroot _macos_target _fc _pixi_fc _fc_driver

    _pixi_fc="${PIXI_ENV}/bin/arm64-apple-darwin20.0.0-gfortran"
    if [ -n "${OMPI_FC}" ] && [ -x "${OMPI_FC}" ]; then
        _fc="${OMPI_FC}"
    elif [ -x "/opt/homebrew/bin/gfortran" ]; then
        _fc="/opt/homebrew/bin/gfortran"
    elif command -v gfortran >/dev/null 2>&1 && [ "$(command -v gfortran)" != "${_pixi_fc}" ]; then
        _fc="$(command -v gfortran)"
    else
        echo "Error: local macOS OpenMPI PETSc builds require an external gfortran"
        echo "Install one with: brew install gcc"
        exit 1
    fi

    _fc_driver="$(
        env -u SDKROOT -u MACOSX_DEPLOYMENT_TARGET \
            "${_fc}" -### -x f95 /dev/null 2>&1 || true
    )"
    _sdkroot="$(
        printf '%s\n' "${_fc_driver}" | python3 -c 'import re, sys; text = sys.stdin.read(); match = re.search(r"-syslibroot\s+(/[^ ]*MacOSX[0-9.]*\.sdk)/?", text); print(match.group(1) if match else "")'
    )"
    _macos_target="$(
        printf '%s\n' "${_fc_driver}" | python3 -c 'import re, sys; text = sys.stdin.read(); match = re.search(r"-mmacosx-version-min=([0-9]+\.[0-9]+)", text); print(match.group(1) if match else "")'
    )"
    if [ -z "${_sdkroot}" ] || [ -z "${_macos_target}" ]; then
        echo "Error: unable to determine gfortran macOS SDK/deployment target"
        exit 1
    fi

    # Strip pixi/conda toolchain env so the MPI wrappers are driven by one
    # coherent compiler stack rather than a mix of conda wrappers and Apple tools.
    unset AS AR CC CXX FC F77 F90 CPP LD NM RANLIB STRIP
    unset CFLAGS CXXFLAGS FFLAGS FORTRANFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD
    unset SDKROOT MACOSX_DEPLOYMENT_TARGET LIBRARY_PATH CPATH
    unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH
    unset CMAKE_ARGS CC_FOR_BUILD CXX_FOR_BUILD FC_FOR_BUILD CPP_FOR_BUILD
    unset OBJC OBJC_FOR_BUILD CLANG CLANGXX GFORTRAN

    export SDKROOT="${_sdkroot}"
    export OMPI_CC=/usr/bin/clang
    export OMPI_CXX=/usr/bin/clang++
    export OMPI_FC="${_fc}"
    export OMPI_CFLAGS="--sysroot=${_sdkroot} -mmacosx-version-min=${_macos_target}"
    export OMPI_CXXFLAGS="--sysroot=${_sdkroot} -mmacosx-version-min=${_macos_target}"
    export OMPI_FCFLAGS="--sysroot=${_sdkroot} -mmacosx-version-min=${_macos_target} -I${PIXI_ENV}/include"

    echo "Applying macOS OpenMPI toolchain overrides:"
    echo "  SDKROOT: ${SDKROOT}"
    echo "  CC/CXX:  ${OMPI_CC} / ${OMPI_CXX}"
    echo "  FC:      ${OMPI_FC}"
    echo "  Target:  macOS ${_macos_target}"
}

clone_petsc() {
    # For Gadi: resolve symlink before cloning. git clone replaces a
    # symlink-to-empty-dir with a real directory, defeating the
    # gdata→scratch symlink approach used to avoid inode quota limits.
    local _clone_target="${PETSC_DIR}"
    if [ "${CLUSTER}" = "gadi" ] && [ -L "${PETSC_DIR}" ]; then
        _clone_target="$(readlink -f "${PETSC_DIR}")"
    fi

    if [ -f "${_clone_target}/configure" ]; then
        echo "PETSc directory already exists. Skipping clone."
        echo "To force fresh clone, run: $0 clean-all"
        return 0
    fi

    echo "Cloning PETSc release branch..."
    git clone -b release https://gitlab.com/petsc/petsc.git "${_clone_target}"
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
    echo "Configuring PETSc with AMR tools (${CLUSTER})..."
    cd "$PETSC_DIR"

    # Capture pixi's python3 BEFORE setup_gadi_build_env reorders PATH.
    local _python="python3"
    if [ "${CLUSTER}" = "gadi" ]; then
        _python="$(which python3)"
        setup_gadi_build_env
    elif [ "${CLUSTER}" = "local" ] && [ "${MPI_IMPL}" = "openmpi" ]; then
        setup_local_macos_openmpi_build_env
    fi

    # Flags shared across all clusters.
    # Downloads and builds:
    #   AMR:          mmg, parmmg, pragmatic, eigen
    #   Solvers:      mumps, scalapack, slepc
    #   Partitioners: metis, parmetis, ptscotch (patched for C23)
    # Uses pixi env (local):  MPI (amr/amr-mpich/amr-openmpi), HDF5
    # Downloads (kaiju):      HDF5, BLAS/LAPACK, cmake, bison
    # Uses module (gadi):     MPI (openmpi/4.1.7), HDF5 ($HDF5_DIR)
    local -a _common=(
        --with-petsc-arch="${PETSC_ARCH}"
        --with-debugging=0
        --with-pragmatic=1
        --with-x=0
        --download-eigen=1
        --download-metis=1
        --download-mmg=1
        "--download-mmg-cmake-arguments=-DMMG_INSTALL_PRIVATE_HEADERS=ON -DUSE_SCOTCH=OFF"
        --download-mumps=1
        --download-parmetis=1
        --download-parmmg=1
        --download-pragmatic=1
        "--download-ptscotch=${SCRIPT_DIR}/patches/scotch-7.0.10-c23-fix.tar.gz"
        --download-scalapack=1
        --download-slepc=1
    )

    case "${CLUSTER}" in
        local)
            "${_python}" ./configure "${_common[@]}" \
                --with-cc="${PIXI_ENV}/bin/mpicc" \
                --with-cxx="${PIXI_ENV}/bin/mpicxx" \
                --with-fc="${PIXI_ENV}/bin/mpif90" \
                --with-hdf5=1 \
                --with-hdf5-dir="${PIXI_ENV}" \
                --download-hdf5=0 \
                --download-mpich=0 \
                --download-openmpi=0 \
                --download-mpi4py=0 \
                --download-bison \
                --with-petsc4py=0
            ;;
        kaiju)
            "${_python}" ./configure "${_common[@]}" \
                --with-mpi-dir="${MPI_DIR}" \
                --download-hdf5=1 \
                --download-fblaslapack=1 \
                --download-cmake=1 \
                --download-bison=1 \
                --with-petsc4py=1 \
		--with-slepc4py=1 \
                --with-make-np=40
            ;;
        gadi)
            "${_python}" ./configure "${_common[@]}" \
                --with-cc="${MPI_DIR}/bin/mpicc" \
                --with-cxx="${MPI_DIR}/bin/mpicxx" \
                --with-fc="${MPI_DIR}/bin/mpifort" \
                --with-hdf5=1 \
                --with-hdf5-dir="${HDF5_DIR}" \
                --download-fblaslapack=1 \
                --with-petsc4py=1 \
		--with-slepc4py=1 \
                --with-make-np=40 \
                --with-shared-libraries=1 \
                --with-cxx-dialect=C++11 \
                "--COPTFLAGS=-g -O3" "--CXXOPTFLAGS=-g -O3" "--FOPTFLAGS=-g -O3" \
                --useThreads=0 \
                --download-zlib=1 \
                --download-superlu=1 \
                --download-superlu_dist=1 \
                --download-hypre=1 \
                --download-ctetgen=1 \
                --download-triangle=1
            ;;
    esac

    echo "Configure complete."
}

build_petsc() {
    echo "Building PETSc (${CLUSTER})..."
    cd "$PETSC_DIR"

    export PETSC_DIR
    export PETSC_ARCH

    if [ "${CLUSTER}" = "gadi" ]; then
        setup_gadi_build_env
    elif [ "${CLUSTER}" = "local" ] && [ "${MPI_IMPL}" = "openmpi" ]; then
        setup_local_macos_openmpi_build_env
    fi

    make all
    echo "PETSc build complete."
}

test_petsc() {
    echo "Testing PETSc..."
    cd "$PETSC_DIR"

    export PETSC_DIR
    export PETSC_ARCH

    if [ "${CLUSTER}" = "gadi" ]; then
        setup_gadi_build_env
    elif [ "${CLUSTER}" = "local" ] && [ "${MPI_IMPL}" = "openmpi" ]; then
        setup_local_macos_openmpi_build_env
    fi

    make check
    echo "PETSc tests complete."
}

build_petsc4py() {
    if [ "${CLUSTER}" != "local" ]; then
        echo "Note: petsc4py is built during configure on HPC clusters (--with-petsc4py=1). Skipping."
        return 0
    fi

    echo "Building petsc4py..."
    cd "$PETSC_DIR/src/binding/petsc4py"

    export PETSC_DIR
    export PETSC_ARCH

    if [ "${CLUSTER}" = "local" ] && [ "${MPI_IMPL}" = "openmpi" ]; then
        setup_local_macos_openmpi_build_env
    fi

    python setup.py build
    python setup.py install
    echo "petsc4py build complete."
}

clean_petsc() {
    local arch_dir="$PETSC_DIR/$PETSC_ARCH"
    echo "Removing PETSc build for $PETSC_ARCH ($arch_dir)..."
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

checkout_version() {
    # Checkout a specific PETSc version tag and update .petsc-version
    local tag="$1"
    if [ -z "${tag}" ]; then
        echo "Usage: $0 checkout <version-tag>"
        echo "  e.g.: $0 checkout v3.25.0"
        echo ""
        echo "Available tags:"
        cd "${PETSC_DIR}" && git tag --list 'v3.2[0-9]*' | sort -V | tail -10
        exit 1
    fi

    cd "${PETSC_DIR}"

    # Fetch latest tags
    git fetch --tags 2>/dev/null

    if ! git rev-parse "${tag}" &>/dev/null; then
        echo "Error: tag '${tag}' not found."
        echo "Available tags:"
        git tag --list 'v3.2[0-9]*' | sort -V | tail -10
        exit 1
    fi

    echo "Checking out PETSc ${tag}..."
    git checkout "${tag}"

    # Update .petsc-version file
    local ver_short
    ver_short=$(echo "${tag}" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\1\2/')
    echo "${ver_short}" > "${VERSION_FILE}"
    echo "Active PETSc version: ${ver_short} (${tag})"
    echo "  Stored in: ${VERSION_FILE}"

    # Check if a build exists for this version
    local new_arch="petsc-${ver_short}-uw-${MPI_IMPL}"
    if [ -d "${PETSC_DIR}/${new_arch}/lib" ]; then
        echo "  Existing build found: ${new_arch}"
        echo "  Run: ./uw build  (to rebuild petsc4py + UW3)"
    else
        echo "  No build for ${new_arch} yet."
        echo "  Run: pixi run -e <env> ./petsc-custom/build-petsc.sh"
    fi
}

list_versions() {
    echo "PETSc version builds:"
    echo ""
    local active_ver
    active_ver=$(petsc_version_short)
    for arch_dir in "${PETSC_DIR}"/petsc-*-uw-*/; do
        [ -d "${arch_dir}" ] || continue
        local arch_name=$(basename "${arch_dir}")
        local marker=""
        local style=""

        # Classify: versioned (petsc-324-uw-*) vs legacy (petsc-4-uw-*)
        if echo "${arch_name}" | grep -qE '^petsc-[0-9]{3,}-uw-'; then
            # Versioned arch: petsc-324-uw-openmpi → 324
            local arch_ver=$(echo "${arch_name}" | sed -E 's/^petsc-([0-9]+)-uw-.*/\1/')
            if [ "${arch_ver}" = "${active_ver}" ]; then
                marker=" (active)"
            fi
        else
            style="  [legacy]"
        fi

        local has_lib=""
        if [ -f "${arch_dir}/lib/libpetsc.dylib" ] || [ -f "${arch_dir}/lib/libpetsc.so" ]; then
            has_lib="  [built]"
        else
            has_lib="  [incomplete]"
        fi
        echo "  ${arch_name}${has_lib}${style}${marker}"
    done
    echo ""
    if [ -d "${PETSC_DIR}/.git" ]; then
        local tag
        tag=$(cd "${PETSC_DIR}" && git describe --tags --abbrev=0 2>/dev/null || echo "unknown")
        echo "Source checkout: ${tag}"
    fi
    echo "Active version: ${active_ver} (from ${VERSION_FILE})"
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Cluster: ${CLUSTER}  (override: export UW_CLUSTER=local|kaiju|gadi)"
    echo "PETSC_ARCH: ${PETSC_ARCH}  (version: ${PETSC_VER})"
    echo ""
    echo "Commands:"
    echo "  (none)      Full build: clone, patch, configure, build"
    [ "${CLUSTER}" = "local" ] && echo "              (local: also runs petsc4py separately)"
    echo "  clone       Clone PETSc repository"
    echo "  checkout V  Checkout PETSc version tag (e.g. v3.25.0) and set active"
    echo "  patch       Apply UW3 patches to PETSc source"
    echo "  configure   Configure PETSc with AMR tools"
    echo "  build       Build PETSc"
    echo "  test        Run PETSc tests"
    echo "  petsc4py    Build and install petsc4py (local only)"
    echo "  versions    List available PETSc builds"
    echo "  clean       Remove build for current arch (${PETSC_ARCH})"
    echo "  clean-all   Remove entire PETSc directory (all builds)"
    echo "  help        Show this help"
    if [ "${CLUSTER}" = "local" ]; then
        echo ""
        echo "Version switching:"
        echo "  $0 checkout v3.25.0   # switch source to 3.25.0"
        echo "  $0                     # build (creates petsc-325-uw-openmpi)"
        echo "  $0 checkout v3.24.2   # switch back (existing build reused)"
        echo ""
        echo "MPI variants co-exist. To build both:"
        echo "  pixi run -e amr         ./petsc-custom/build-petsc.sh"
        echo "  pixi run -e amr-openmpi ./petsc-custom/build-petsc.sh"
    fi
}

# ── Main entry point ──────────────────────────────────────────────────────────
case "${1:-all}" in
    all)
        clone_petsc
        apply_patches
        configure_petsc
        build_petsc
        if [ "${CLUSTER}" = "local" ]; then
            build_petsc4py
        fi
        echo ""
        echo "=========================================="
        echo "PETSc AMR build complete! (${CLUSTER}, ${MPI_IMPL})"
        echo "  PETSC_DIR=${PETSC_DIR}"
        echo "  PETSC_ARCH=${PETSC_ARCH}"
        if [ "${CLUSTER}" != "local" ]; then
            echo "  export PYTHONPATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$PYTHONPATH"
        fi
        echo "=========================================="
        ;;
    clone)     clone_petsc ;;
    checkout)  checkout_version "$2" ;;
    patch)     apply_patches ;;
    configure) configure_petsc ;;
    build)     build_petsc ;;
    test)      test_petsc ;;
    petsc4py)  build_petsc4py ;;
    versions)  list_versions ;;
    clean)     clean_petsc ;;
    clean-all) clean_all ;;
    help|--help|-h) show_help ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
