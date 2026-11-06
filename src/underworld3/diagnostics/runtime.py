"""MPI-aware runtime information and shared-file hashing for UW3 clients."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shlex
import sys
from mpi4py import MPI


def collective_file_sha256(path: Path, comm: MPI.Comm) -> str:
    """Stream a shared file once on rank zero and broadcast its hash or error.

    Meshes at fine spherical resolutions can be gigabytes. Reading them into
    memory on every rank just for provenance multiplies memory and I/O costs.
    Every communicator rank must enter this call with the same shared file.
    """
    digest, error = None, None
    if comm.rank == 0:
        try:
            with Path(path).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
    digest, error = comm.bcast((digest, error), root=0)
    if error is not None:
        raise OSError(f"Shared provenance hash failed for {path}: {error}")
    return digest


def runtime_info(*, comm: MPI.Comm) -> dict[str, object]:
    """Collect package/host/MPI information and maximum per-rank lifetime RSS.

    Every communicator rank must call. Source-checkout and campaign provenance
    belong to the application; these values describe the running process.
    """
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "sympy",
        "h5py",
        "mpi4py",
        "petsc4py",
        "gmsh",
        "underworld3",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    peak_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB. Reduce only after normalising.
    peak_bytes = peak_raw if sys.platform == "darwin" else 1024.0 * peak_raw
    peak_bytes_rank_max = float(comm.allreduce(peak_bytes, op=MPI.MAX))
    return {
        "command_line": shlex.join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "mpi_library": MPI.Get_library_version().replace("\x00", ""),
        "package_versions_json": json.dumps(packages, sort_keys=True),
        "peak_resident_set_size_bytes_rank_max": peak_bytes_rank_max,
    }
