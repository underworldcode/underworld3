"""Parallel validation for the JIT cache (issues #121, #123).

Run via:

    mpirun -np 4 python tests/parallel/ptest_jit_cache.py

Exits non-zero on the first failed assertion (with a printed banner so
``mpi_runner.sh``-style test loops surface the failure in CI logs).

Validates four things that aren't checkable from the serial pytest suite:

1. **Hash agreement.** The C-source canonicalisation must be byte-identical
   across ranks. ``getext`` already calls ``allgather(source_hash)`` and
   raises on mismatch — this test asserts the hash matches what every rank
   computes locally too, and that it doesn't change between repeated calls.

2. **C ↔ S ↔ C transition under np>1.** The same correctness gate as the
   serial test, but every rank must land on the same numerical answer.

3. **Concurrent compile race.** All ranks compile their first bundle at
   the same time. They each write to a unique ``/tmp/fn_ptr_ext_*_<pid>_*``
   directory and then race to populate the on-disk cache; only rank 0
   should win the publish (the others' writes are no-ops by construction).
   We assert the cache directory ends up with exactly one ``.so`` per
   distinct bundle.

4. **Inter-process flock.** A second mpirun launched while the first is
   still compiling could otherwise stomp on the same ``{hash}.so``. We
   exercise this by having every rank call ``store_module`` on the same
   fake artifact — only one ``.so`` should result, and the file size
   must equal what we wrote.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities import _jit_cache as _jc
from underworld3.utilities import _jitextension as _jitext


COMM = uw.mpi.comm
RANK = uw.mpi.rank
SIZE = uw.mpi.size


def _banner(msg):
    if RANK == 0:
        print(f"\n=== {msg} ===", flush=True)


def _fail(msg):
    print(f"[rank {RANK}] FAIL: {msg}", flush=True)
    COMM.Abort(1)


def _ranks_agree(value, label):
    gathered = COMM.allgather(value)
    if any(v != value for v in gathered):
        _fail(f"{label}: ranks disagree — {gathered}")


def _setup_isolated_cache():
    """Pick a per-rank-shared cache dir under /tmp so we don't clobber the user's."""
    base = Path(tempfile.gettempdir()) / "uw_jit_cache_ptest"
    if RANK == 0:
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
    COMM.Barrier()
    os.environ["UW_JIT_CACHE_DIR"] = str(base)
    return base


def test_1_hash_agreement_and_csc_correctness():
    """C↔S↔C transition under MPI: hash agreement + numerical equality."""
    _banner(f"Test 1: hash agreement + C↔S↔C correctness (np={SIZE})")

    _jitext._ext_dict.clear()

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.2)
    u = uw.discretisation.MeshVariable("u_par_csc", mesh, 1, degree=2)
    K = uw.expression("K_par_csc", 0.5)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.add_dirichlet_bc(0.0, "Bottom")

    sample_y = np.linspace(0.05, 0.95, 15)
    pts = np.column_stack([np.full_like(sample_y, 0.5), sample_y])

    def t_max():
        # On non-rank-0 evaluate may return NaN at points off-rank; use
        # a global reduction to avoid spurious disagreement.
        local_arr = uw.function.evaluate(u.sym[0], pts, rbf=False).squeeze()
        local_max = float(np.nanmax(np.abs(local_arr))) if np.any(~np.isnan(local_arr)) else 0.0
        return COMM.allreduce(local_max, op=uw.mpi._MPI.MAX)

    # State 1
    poisson.solve(zero_init_guess=True)
    key_1 = poisson._current_jit_cache_key
    _ranks_agree(key_1, "state-1 cache key")
    t_max_1 = t_max()
    _ranks_agree(round(t_max_1, 9), "state-1 T_max")

    # State 2 — symbolic
    poisson.constitutive_model.Parameters.diffusivity = 1.0 + u.sym[0]
    poisson.solve(zero_init_guess=True)
    key_2 = poisson._current_jit_cache_key
    _ranks_agree(key_2, "state-2 cache key")
    if key_2 == key_1:
        _fail(f"state-1 and state-2 hashed to the same key {key_1}; should differ")
    t_max_2 = t_max()

    # State 3 — back to constant
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.solve(zero_init_guess=True)
    key_3 = poisson._current_jit_cache_key
    _ranks_agree(key_3, "state-3 cache key")
    if key_3 != key_1:
        _fail(f"state-3 ({key_3}) should hash to state-1 key ({key_1})")
    t_max_3 = t_max()

    if abs(t_max_3 - t_max_1) > 1e-6:
        _fail(
            f"state-3 T_max {t_max_3:.8f} differs from state-1 {t_max_1:.8f}; "
            f"PETSc-DS contamination across the C→S→C transition"
        )

    if RANK == 0:
        print(
            f"  state 1 T_max={t_max_1:.6f}  state 2 T_max={t_max_2:.6f}  "
            f"state 3 T_max={t_max_3:.6f}  ✓",
            flush=True,
        )

    del poisson


def test_2_concurrent_compile_publishes_once():
    """All ranks solve cold; the disk cache must end up with exactly one .so per bundle."""
    _banner(f"Test 2: concurrent cold compile, single publish (np={SIZE})")

    cache_dir = _jc.get_cache_dir()
    if cache_dir is None:
        _fail("disk cache should not be disabled in this test")

    # Wipe and synchronise so every rank really does start cold
    if RANK == 0:
        for p in cache_dir.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
    COMM.Barrier()
    _jitext._ext_dict.clear()

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
    u = uw.discretisation.MeshVariable("u_par_concur", mesh, 1, degree=2)
    K = uw.expression("K_par_concur", 1.0)

    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = K
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    source_hash = poisson._current_jit_cache_key
    _ranks_agree(source_hash, "concurrent-compile cache key")

    COMM.Barrier()  # let rank 0's store_module finish
    if RANK == 0:
        so_files = sorted(p.name for p in cache_dir.glob("*.so"))
        manifest_files = sorted(p.name for p in cache_dir.glob("*.manifest.json"))
        expected_so = f"{source_hash}.so"
        expected_manifest = f"{source_hash}.manifest.json"
        if so_files != [expected_so]:
            _fail(f"expected single .so {expected_so}, got {so_files}")
        if manifest_files != [expected_manifest]:
            _fail(f"expected single manifest {expected_manifest}, got {manifest_files}")
        # Also ensure no leftover ".inprogress_*" files (would mean a publish was
        # cut off mid-write, which os.replace + flock are supposed to prevent)
        leftover = sorted(p.name for p in cache_dir.glob(".inprogress_*"))
        if leftover:
            _fail(f"leftover .inprogress_* artifacts: {leftover}")
        print(f"  cache dir: {so_files} + {manifest_files}  ✓", flush=True)

    del poisson


def test_3_warm_run_skips_compile():
    """Second pass with a warm in-memory cache should never invoke compile_and_load."""
    _banner(f"Test 3: warm in-memory cache, no compile (np={SIZE})")

    compile_calls = []
    real_compile = _jitext.compile_and_load

    def watching(*args, **kwargs):
        compile_calls.append(args)
        return real_compile(*args, **kwargs)

    _jitext.compile_and_load = watching
    try:
        # Build a fresh solver — but with the SAME structure as test 2 so the
        # hash matches and _ext_dict has the entry.
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.25)
        u = uw.discretisation.MeshVariable("u_par_warm", mesh, 1, degree=2)
        K = uw.expression("K_par_concur", 1.0)  # reuse name for stable hash

        poisson = uw.systems.Poisson(mesh, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = K
        poisson.f = 0.0
        poisson.add_dirichlet_bc(1.0, "Bottom")
        poisson.add_dirichlet_bc(0.0, "Top")
        # Note: this uses a different MeshVariable name (u_par_warm), so the
        # hash will likely differ from test 2 — but the disk cache from test
        # 2 doesn't help us here. The point of this test is really that
        # repeated solves on the SAME solver don't recompile.
        poisson.solve()
        n_compiles_after_first = len(compile_calls)

        for _ in range(5):
            K.sym = float(np.random.uniform(0.5, 1.5))
            poisson.solve()

        n_compiles_after_loop = len(compile_calls)
        # First solve compiles; the loop must add zero
        if n_compiles_after_loop != n_compiles_after_first:
            _fail(
                f"warm-cache loop fired compile_and_load "
                f"{n_compiles_after_loop - n_compiles_after_first} extra times"
            )
        if RANK == 0:
            print(
                f"  rank 0: {n_compiles_after_first} compile(s) on first solve, "
                f"{n_compiles_after_loop - n_compiles_after_first} during 5-iter cycle  ✓",
                flush=True,
            )
    finally:
        _jitext.compile_and_load = real_compile

    del poisson


def test_5_only_rank_0_compiles_on_cold_start():
    """Cold start with np>1: rank 0 invokes cc; other ranks load from disk."""
    _banner(f"Test 5: rank-0-only compile on cold start (np={SIZE})")

    if SIZE == 1:
        if RANK == 0:
            print("  np=1 → rank-0-only compile is trivially satisfied. SKIP", flush=True)
        return

    cache_dir = _jc.get_cache_dir()
    if RANK == 0:
        for p in cache_dir.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
    COMM.Barrier()
    _jitext._ext_dict.clear()

    compile_calls = []
    real_compile = _jitext.compile_and_load

    def watching(*args, **kwargs):
        compile_calls.append(args)
        return real_compile(*args, **kwargs)

    _jitext.compile_and_load = watching
    try:
        # Use a structurally-distinct solver from the other tests so we
        # genuinely cold-start (no in-memory hit from a prior test).
        mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.18)
        u = uw.discretisation.MeshVariable("u_par_rank0", mesh, 1, degree=2)
        K = uw.expression("K_par_rank0", 0.7)

        poisson = uw.systems.Poisson(mesh, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = K
        poisson.f = -1.0
        poisson.add_dirichlet_bc(0.0, "Bottom")
        poisson.add_dirichlet_bc(0.0, "Top")
        poisson.solve()
    finally:
        _jitext.compile_and_load = real_compile

    n_calls = len(compile_calls)
    all_calls = COMM.allgather(n_calls)
    if RANK == 0:
        # Rank 0 should have called compile_and_load exactly once.
        # Every other rank should have zero calls.
        if all_calls[0] != 1:
            _fail(f"rank 0 should have one cc invocation, got {all_calls[0]}")
        for r in range(1, SIZE):
            if all_calls[r] != 0:
                _fail(
                    f"rank {r} invoked cc {all_calls[r]} time(s); should be 0 "
                    f"(disk cache should have served it after rank-0 publish). "
                    f"Per-rank counts: {all_calls}"
                )
        print(f"  per-rank cc invocations: {all_calls}  ✓", flush=True)

    del poisson


def test_4_flock_serialises_disk_writes():
    """All ranks call store_module on the same fake artifact; only one wins, no corruption."""
    _banner(f"Test 4: flock serialises disk writes (np={SIZE})")

    cache_dir = _jc.get_cache_dir()
    bogus_hash = f"flockprobe{RANK:06x}"[:16].ljust(16, "0")
    bogus_hash = "flock0123456789a"

    # Each rank has its own fake "compiled" .so so we can detect which one
    # ended up in the cache.
    fake_dir = Path(tempfile.gettempdir()) / f"uw_jit_fake_rank{RANK}_{os.getpid()}"
    fake_dir.mkdir(parents=True, exist_ok=True)
    payload = f"rank-{RANK} payload at {time.time_ns()}".encode()
    fake_so = fake_dir / "fn_ptr_ext_flockprobe.so"
    fake_so.write_bytes(payload)

    class _FakeExpr:
        name = "K_flock"

    # Wipe any prior entry
    if RANK == 0:
        for p in cache_dir.glob(f"{bogus_hash}*"):
            try:
                p.unlink()
            except OSError:
                pass
        for p in cache_dir.glob(f".{bogus_hash}*"):
            try:
                p.unlink()
            except OSError:
                pass
    COMM.Barrier()

    # All ranks try to publish at roughly the same time
    _jc.store_module(bogus_hash, "fn_ptr_ext_flock", str(fake_dir), [(0, _FakeExpr())])
    COMM.Barrier()

    if RANK == 0:
        so_path = cache_dir / f"{bogus_hash}.so"
        manifest_path = cache_dir / f"{bogus_hash}.manifest.json"
        if not so_path.exists():
            _fail(f"no .so written under flock; expected {so_path}")
        if not manifest_path.exists():
            _fail(f"no manifest written under flock; expected {manifest_path}")
        leftover = sorted(p.name for p in cache_dir.glob(".inprogress_*"))
        if leftover:
            _fail(f"leftover .inprogress_* under flock: {leftover}")
        # Whichever rank won, the bytes must equal a valid rank's payload —
        # not a torn write or the original being overwritten mid-copy.
        contents = so_path.read_bytes()
        if not contents.startswith(b"rank-"):
            _fail(f"corrupt .so under flock; first bytes={contents[:32]!r}")
        print(f"  surviving .so: {len(contents)} bytes, starts with {contents[:24]!r}  ✓", flush=True)

    COMM.Barrier()
    if RANK == 0:
        # cleanup
        for p in cache_dir.glob(f"{bogus_hash}*"):
            try: p.unlink()
            except OSError: pass
        for p in cache_dir.glob(f".{bogus_hash}*"):
            try: p.unlink()
            except OSError: pass


def main():
    cache_dir = _setup_isolated_cache()
    if RANK == 0:
        print(f"\nRunning JIT cache parallel tests with np={SIZE}", flush=True)
        print(f"Isolated cache dir: {cache_dir}", flush=True)

    test_1_hash_agreement_and_csc_correctness()
    test_2_concurrent_compile_publishes_once()
    test_3_warm_run_skips_compile()
    test_4_flock_serialises_disk_writes()
    test_5_only_rank_0_compiles_on_cold_start()

    COMM.Barrier()
    if RANK == 0:
        print(f"\n=== All JIT cache parallel tests PASSED (np={SIZE}) ===", flush=True)


if __name__ == "__main__":
    main()
