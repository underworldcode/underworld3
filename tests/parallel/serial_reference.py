"""Compute a parallel test's OWN diagnostic at np=1, in THIS environment, so the
partition-independence assertion compares two runs of the same code on the same host.

Why this exists
---------------
The partition tests used to compare against a constant recorded on a developer's
machine, and their failure messages said "differs serial vs np=N" when what they
actually measured was "differs from a number recorded elsewhere". Those are not the
same statement, and conflating them cost a day of investigation: seven assertions
failed on CI, were read as a partition-dependence family, and four of them turned out
to be the mesh. Running ``test_1064``'s own annulus diagnostic at both rank counts on
ONE CI host gives np=1 ``1.897329151623790e-02`` and np=2 ``1.897329151623740e-02`` —
agreement to the 13th significant figure — while BOTH differ from the recorded golden
by the same +1.676e-04, because gmsh builds a different triangulation on the Linux
runner and the tests are ``mpi(min_size=2)`` so CI never ran np=1 to notice.

A self-referential comparison is immune to that: whatever mesh the host generates, both
sides of the comparison use it.

How
---
Rank 0 spawns a plain single-rank Python running the test module's own ``__main__``,
which prints one ``SERIALREF <json>`` line; the payload is broadcast to every rank. The
child's environment is scrubbed of the launcher's MPI variables — inherited ``OMPI_*`` /
``PMIX_*`` make the child believe it is a member of the parent's job, and it then hangs
or aborts instead of running as a singleton.

The mesh is read from the same gmsh cache the parent uses, so parent and child are the
same triangulation by construction; the fingerprint is carried through anyway and
reported on failure, so the day that stops being true it says so.

COLLECTIVE — every rank must call :func:`serial_reference` (rank 0 runs the child, the
others wait in the broadcast).
"""
import json
import os
import subprocess
import sys

import numpy as np

import underworld3 as uw

# Launcher variables that would make a spawned singleton try to join the parent job.
_MPI_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "MPI_", "HYDRA_", "I_MPI_", "SLURM_")

_CACHE = {}


def mesh_fingerprint(mesh):
    """A partition-independent identity for the mesh a diagnostic ran on: ``(global
    cell count, ∫1 dV)``. A different triangulation moves both (the discretised volume
    of a curved domain moves ~0.1 % between triangulations); a different partition of
    the same triangulation moves neither.

    OWNED cells only. The overlap layer puts the same cell on more than one rank, so a
    plain sum of the local cell-stratum sizes grows with the rank count and would make
    the fingerprint report a partition as if it were a mesh change."""
    import sympy

    dm = mesh.dm
    ghosts = set()
    if uw.mpi.size > 1:
        _nroots, local, _remote = dm.getPointSF().getGraph()
        if local is not None:
            ghosts = set(int(point) for point in local)
    cell_start, cell_end = dm.getHeightStratum(0)
    owned = sum(1 for c in range(cell_start, cell_end) if c not in ghosts)
    cells = int(uw.mpi.comm.allreduce(owned))
    volume = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    return [cells, volume]


def serial_reference(module_file, kind, timeout=600):
    """Run ``python <module_file> <kind>`` as a single-rank child and return the JSON
    payload it printed on its ``SERIALREF`` line. Cached per (module, kind) within the
    process. Raises with the child's output if it did not produce one.

    COLLECTIVE. Rank 0 runs the child, everyone waits in the broadcast — so rank 0
    must reach that broadcast on EVERY path. ``_run_child`` therefore catches
    everything and returns the failure as a string rather than raising: an
    ``OSError`` from ``subprocess.run``, a truncated ``SERIALREF`` line, a
    ``MemoryError`` on ``capture_output`` would otherwise leave rank 0 unwinding out
    of here while every other rank sits in ``MPI_Bcast`` (busy-polling, a core each)
    until pytest-timeout fires.

    The default timeout is deliberately SHORTER than the ``pytest.mark.timeout`` the
    test files carry (600 s vs 900 s), so a stuck child is reported as a stuck child
    instead of being overtaken by the outer timeout and reported as a stuck test.
    """
    key = (os.path.abspath(module_file), kind)
    if key in _CACHE:
        return _CACHE[key]

    payload = _run_child(module_file, kind, timeout) if uw.mpi.rank == 0 else None
    payload = uw.mpi.comm.bcast(payload, root=0)
    if isinstance(payload, str):
        raise RuntimeError(payload)
    _CACHE[key] = payload
    return payload


def _run_child(module_file, kind, timeout):
    """Never raises — see :func:`serial_reference`. Returns the payload dict, or a
    string describing what went wrong."""
    name = os.path.basename(module_file)
    try:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(_MPI_ENV_PREFIXES)}
        proc = subprocess.run(
            [sys.executable, "-u", os.path.abspath(module_file), kind],
            env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"serial reference for {name}:{kind} timed out after {timeout}s"
    except Exception as exc:                     # noqa: BLE001 - see the docstring
        return (f"serial reference for {name}:{kind} could not be launched — "
                f"{type(exc).__name__}: {exc}")
    try:
        for line in proc.stdout.splitlines():
            if line.startswith("SERIALREF "):
                return json.loads(line[len("SERIALREF "):])
    except Exception as exc:                     # noqa: BLE001 - see the docstring
        return (f"serial reference for {name}:{kind} printed an unreadable "
                f"SERIALREF line — {type(exc).__name__}: {exc}")
    return (f"serial reference for {name}:{kind} printed no "
            f"SERIALREF line (rc={proc.returncode})\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}")


def emit(values, fingerprint):
    """Print the ``SERIALREF`` line a test module's ``__main__`` owes its parallel
    twin. Rank-0 only, so it is safe to call unconditionally."""
    if uw.mpi.rank == 0:
        print("SERIALREF " + json.dumps(
            {"values": [float(v) for v in np.atleast_1d(values)],
             "fingerprint": [float(f) for f in fingerprint]}))


def _same_mesh(left, right):
    """Do two fingerprints describe the same triangulation? Cell count exactly, volume
    to 1e-12 relative (it is a sum of the same element volumes in a partition-dependent
    ORDER, so the last couple of bits move; a different triangulation moves it by ~1e-3
    relative, nine orders away)."""
    return (int(left[0]) == int(right[0])
            and np.isclose(left[1], right[1], rtol=1e-12, atol=0))


def _fp(fingerprint):
    return f"cells={fingerprint[0]:.0f} vol={fingerprint[1]:.12g}"


def compare(values, reference, rtols, labels, fingerprint, what):
    """Assert each of ``values`` matches the serial reference within its ``rtols``,
    and say what moved — including both mesh fingerprints, so a host/mesh difference
    reads as a mesh difference instead of as a physics regression.

    The fingerprints are ASSERTED equal, not merely reported. The np=1 child reads the
    same gmsh cache as the parallel parent, so a mismatch means something has broken
    that assumption (a concurrent run regenerating the cache, a fingerprint that is not
    partition-independent after all) and every number below it would be meaningless.
    """
    ref_values = reference["values"]
    ref_fp = reference["fingerprint"]
    assert len(values) == len(ref_values), (
        f"{what}: serial reference has {len(ref_values)} values, this run produced "
        f"{len(values)}")
    assert _same_mesh(fingerprint, ref_fp), (
        f"{what}: the np=1 reference ran on a DIFFERENT mesh from this np={uw.mpi.size} "
        f"run — np=1 [{_fp(ref_fp)}] vs np={uw.mpi.size} [{_fp(fingerprint)}]. The "
        f"comparison below would be measuring the mesh, not the partition.")
    fp_note = f" [mesh {_fp(fingerprint)}]"
    for value, ref, rtol, label in zip(values, ref_values, rtols, labels):
        assert np.isclose(value, ref, rtol=rtol, atol=0), (
            f"{what}: {label} is partition dependent — np=1 {ref!r} vs "
            f"np={uw.mpi.size} {value!r} (rtol {rtol:g}){fp_note}")


def accuracy_anchor(values, anchor, fingerprint, labels, what, rtol=1e-2):
    """Assert the ABSOLUTE answer against a recorded constant, GATED on the mesh.

    Partition independence and accuracy are two different claims and the self-referential
    comparison above only makes the first. A rotated constraint that stopped constraining
    equally on every rank, an FMG hierarchy that converged to the wrong answer, a
    physics benchmark coefficient that drifted — all of those pass ``compare`` and are
    caught only by a number recorded when the result was known good.

    The reason those constants were removed is real: they are host-specific, because
    gmsh triangulates differently on different platforms, and a mismatch then reads as a
    physics regression. The fix is the one the #564 investigation actually recommended —
    keep the constant, and put the mesh fingerprint in front of it. On the host the
    anchor was recorded on this is a live accuracy gate; on any other mesh it SKIPS,
    loudly, instead of failing for the wrong reason.

    ``rtol`` is deliberately loose (1 %). This is not a reproducibility check — that is
    ``compare``'s job, three to eight orders tighter. This one only has to notice that
    the answer has become a different answer.
    """
    import pytest

    if not _same_mesh(fingerprint, anchor["fingerprint"]):
        pytest.skip(
            f"{what}: accuracy anchor was recorded on a different mesh "
            f"[{_fp(anchor['fingerprint'])}] from this host's "
            f"[{_fp(fingerprint)}] — gmsh triangulates differently across "
            f"platforms. Partition independence is still asserted; only the "
            f"absolute value is skipped.")
    for value, ref, label in zip(values, anchor["values"], labels):
        assert np.isclose(value, ref, rtol=rtol, atol=0), (
            f"{what}: {label} has MOVED from its recorded value on the same mesh "
            f"[{_fp(fingerprint)}] — {ref!r} recorded, {value!r} now "
            f"(rtol {rtol:g}). This is an accuracy regression, not a partition "
            f"effect: the mesh is identical and the partition check passed.")
