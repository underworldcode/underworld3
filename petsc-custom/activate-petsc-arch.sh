#!/usr/bin/env bash
# activate-petsc-arch.sh — sourced by pixi on environment activation.
#
# Dynamically sets PETSC_ARCH for AMR environments (custom PETSc builds)
# by combining the active pixi environment name with the PETSc version
# pinned in petsc-custom/.petsc-version.
#
# This replaces hardcoded PETSC_ARCH entries in pixi.toml so that
# switching PETSc versions (via ./uw petsc switch <tag>) does not require
# editing tracked configuration.

_uw_mpi=""
_uw_suffix=""

# Default MPI for AMR envs is platform-dependent and mirrors the conda deps
# pinned in pixi.toml: openmpi on macOS (feature.amr.target.osx-arm64),
# mpich on Linux (feature.amr.target.linux-64).
_uw_default_mpi="mpich"
[ "$(uname -s)" = "Darwin" ] && _uw_default_mpi="openmpi"

case "${PIXI_ENVIRONMENT_NAME:-}" in
    amr|amr-runtime|amr-dev)     _uw_mpi="$_uw_default_mpi" ;;
    amr-mpich|amr-mpich-dev)     _uw_mpi="mpich" ;;
    amr-openmpi|amr-openmpi-dev) _uw_mpi="openmpi" ;;
    amr-debug)                   _uw_mpi="$_uw_default_mpi"; _uw_suffix="-debug" ;;
esac

if [ -n "$_uw_mpi" ]; then
    # Match the version-discovery logic in ./uw (petsc_version_short):
    # prefer .petsc-version, strip whitespace, fall back to 324.
    _uw_ver="324"
    _uw_ver_file="${PIXI_PROJECT_ROOT:-.}/petsc-custom/.petsc-version"
    if [ -f "$_uw_ver_file" ]; then
        _uw_read=$(tr -d '[:space:]' < "$_uw_ver_file" 2>/dev/null)
        [ -n "$_uw_read" ] && _uw_ver="$_uw_read"
    fi

    export PETSC_ARCH="petsc-${_uw_ver}-uw-${_uw_mpi}${_uw_suffix}"
fi

unset _uw_mpi _uw_suffix _uw_default_mpi _uw_ver _uw_ver_file _uw_read
