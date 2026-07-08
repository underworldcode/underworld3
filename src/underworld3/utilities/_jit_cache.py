"""Persistent on-disk cache for JIT-compiled UW3 solver extensions.

Populated by :func:`getext` the first time a bundle is compiled; queried on
subsequent runs (same process or fresh Python process) to skip the Cython
+ cc compile step entirely. Cache entries are keyed by the same SHA-256 of
the canonical C source that drives the in-memory cache — see
``docs/developer/subsystems/jit-cache.md`` for the full design.

Cache directory resolution:

1. ``$UW_JIT_CACHE_DIR`` if set
2. ``$XDG_CACHE_HOME/underworld3/jit`` if ``XDG_CACHE_HOME`` set
3. ``~/.cache/underworld3/jit`` otherwise

The cache is disabled entirely when ``UW_JIT_CACHE=0`` (or ``false``/``no``),
in which case both :func:`load_module` and :func:`store_module` become no-ops.

MPI is **not** handled here — phase 4 of the JIT cache refactor adds rank-0
gating and ``flock``-based inter-process synchronisation in this same module.
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib.machinery
import json
import os
import shutil
import tempfile
from importlib._bootstrap import _load
from pathlib import Path
from typing import Optional


MANIFEST_VERSION = 1


def get_cache_dir() -> Optional[Path]:
    """Return the active JIT cache directory, or ``None`` if disabled."""
    if os.environ.get("UW_JIT_CACHE", "").lower() in ("0", "false", "no"):
        return None
    explicit = os.environ.get("UW_JIT_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "underworld3" / "jit"
    return Path.home() / ".cache" / "underworld3" / "jit"


def _so_path(cache_dir: Path, source_hash: str) -> Path:
    return cache_dir / f"{source_hash}.so"


def _manifest_path(cache_dir: Path, source_hash: str) -> Path:
    return cache_dir / f"{source_hash}.manifest.json"


def _lock_path(cache_dir: Path, source_hash: str) -> Path:
    return cache_dir / f".{source_hash}.lock"


@contextlib.contextmanager
def _file_lock(path: Path):
    """Per-hash advisory lock so two processes don't race on the same entry.

    Uses :func:`fcntl.flock` (POSIX). The lockfile is created if missing
    and kept around — wiping the cache directory removes it cleanly.
    Best-effort: if flock isn't supported (rare on POSIX), the body still
    runs; correctness then relies on the atomic ``os.replace`` in
    :func:`store_module`.
    """
    fd = None
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def _manifest_from_constants(constants_manifest):
    """Canonicalise ``constants_manifest`` for JSON storage / comparison.

    ``constants_manifest`` is a list of ``(index, UWexpression)`` tuples.
    We persist only the stable ``expr.name`` — the current value is irrelevant
    (it's updated via ``PetscDSSetConstants`` at solve time).
    """
    return [{"index": idx, "name": expr.name} for idx, expr in constants_manifest]


def load_module(source_hash: str, modname: str, constants_manifest):
    """Try to load a cached compiled module for ``source_hash``.

    Parameters
    ----------
    source_hash : str
        The hex hash of the canonical C source (same key the in-memory
        cache uses).
    modname : str
        The fully-qualified module name to register with Python's import
        machinery (typically ``fn_ptr_ext_<hash>``).
    constants_manifest : list of (int, UWexpression)
        The current call's constants manifest. Used as a belt-and-braces
        consistency check: the saved manifest on disk must match, otherwise
        we treat it as a cache miss (probable ABI drift).

    Returns
    -------
    module or None
        The dynamically-loaded extension module on hit, ``None`` on miss
        (no ``.so`` on disk, malformed manifest, or manifest mismatch).
    """
    cache_dir = get_cache_dir()
    if cache_dir is None:
        return None
    so_path = _so_path(cache_dir, source_hash)
    manifest_path = _manifest_path(cache_dir, source_hash)
    if not so_path.exists() or not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("version") != MANIFEST_VERSION:
        return None
    saved = manifest.get("constants", [])
    current = _manifest_from_constants(constants_manifest)
    if saved != current:
        # Names shifted between the saved entry and the current call.
        # Could be a UWexpression rename or unnoticed ABI change — safer
        # to recompile than trust a stale mapping.
        return None
    try:
        loader = importlib.machinery.ExtensionFileLoader(modname, str(so_path))
        spec = importlib.machinery.ModuleSpec(
            name=modname, loader=loader, origin=str(so_path)
        )
        return _load(spec)
    except Exception:
        return None


def store_module(source_hash: str, modname: str, tmpdir, constants_manifest) -> None:
    """Copy a freshly-compiled ``.so`` into the cache dir and write its manifest.

    Parameters
    ----------
    source_hash, modname
        See :func:`load_module`.
    tmpdir : str or Path
        The temporary build directory returned by :func:`compile_and_load`
        (contains the ``.so`` we just built).
    constants_manifest : list of (int, UWexpression)
        Persisted as ``{index, name}`` pairs; used by :func:`load_module`
        to verify compatibility on subsequent loads.
    """
    import underworld3

    if underworld3.mpi.rank != 0:
        # Phase 4 will replace this guard with proper MPI locking.
        return
    cache_dir = get_cache_dir()
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    tmp = Path(tmpdir)
    so_files = list(tmp.glob("*.so"))
    if not so_files:
        return
    src_so = so_files[0]
    dst_so = _so_path(cache_dir, source_hash)
    manifest_path = _manifest_path(cache_dir, source_hash)
    lock_path = _lock_path(cache_dir, source_hash)

    # flock prevents a concurrent process (different mpirun, etc.) from
    # interleaving its write with ours. The recheck-inside-lock pattern
    # avoids redundant copies when another process has already populated
    # the entry while we were waiting on the lock.
    with _file_lock(lock_path):
        if dst_so.exists() and manifest_path.exists():
            return

        try:
            tf = tempfile.NamedTemporaryFile(
                dir=str(cache_dir), prefix=".inprogress_", suffix=".so", delete=False
            )
            tf_path = Path(tf.name)
            tf.close()
            shutil.copy2(src_so, tf_path)
            os.replace(tf_path, dst_so)
        except OSError:
            return

        manifest = {
            "version": MANIFEST_VERSION,
            "modname": modname,
            "constants": _manifest_from_constants(constants_manifest),
        }
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(cache_dir),
                prefix=".inprogress_",
                suffix=".json",
                delete=False,
                mode="w",
                encoding="utf-8",
            ) as tf:
                json.dump(manifest, tf)
                tf_path = Path(tf.name)
            os.replace(tf_path, manifest_path)
        except OSError:
            return
