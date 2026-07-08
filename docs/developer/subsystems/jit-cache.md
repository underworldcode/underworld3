# JIT compilation cache

Solvers in underworld3 compile their pointwise residual / Jacobian functions
to native code via Cython + the system `cc`. For a non-trivial solver this
takes seconds (simple Stokes) to minutes (VE Stokes BDF-2 with viscoplastic
flow). The JIT cache eliminates that cost on every call after the first one
within a session, and on every fresh Python process after the first one on a
given machine.

This document describes the on-disk layout, how cache entries are keyed, and
how the cache is invalidated.

## Layout

```
~/.cache/underworld3/jit/
├── .env-fingerprint                    # written by `./uw build`
├── 7e81af3b554b4126.so                 # the compiled extension
├── 7e81af3b554b4126.manifest.json      # constants index → name mapping
├── .7e81af3b554b4126.lock              # advisory lock used during writes
├── 0202220e00e01df3.so
├── 0202220e00e01df3.manifest.json
└── ...
```

Locations honoured (first match wins):

1. `$UW_JIT_CACHE_DIR` if set
2. `$XDG_CACHE_HOME/underworld3/jit` if `XDG_CACHE_HOME` is set
3. `~/.cache/underworld3/jit`

Setting `UW_JIT_CACHE=0` (or `false` / `no`) disables the on-disk cache
entirely — the in-memory dict still works, but nothing is persisted across
processes.

## Cache key

The key is the SHA-256 of the **canonical** generated C source plus an
**ABI salt**, truncated to 16 hex characters.

```python
canonical_source = "\n".join(setup_py, cy_ext_h, cy_ext_pyx)
                   .replace(<modname>, "__UW_JIT_MOD__")
                   .replace(<randstr>, "__UW_JIT_RS__")

salt = f"petsc={PETSc.Sys.getVersion()}|uw={underworld3.__version__}"

source_hash = sha256(canonical_source + "\n---\n" + salt).hexdigest()[:16]
```

Two design points worth understanding:

**Canonicalisation**. The Cython module name and the symbol-prefix randomiser
that the JIT generator picks vary between calls (random or counter-based).
Hashing them directly would make every call produce a fresh key. We instead
hash the source with both replaced by stable placeholders, and only after
hashing do we substitute the real per-bundle names back in. Same C ⇒ same
hash; different C ⇒ different hash; identifiers stay unique per bundle.

**Why hash the C source rather than the sympy callback structure**. Two
sympy expressions can be structurally distinct but emit identical C
(constant placeholders, simplification, etc.); two structurally-identical
expressions can emit different C if a path in the generator depends on
something subtle. The compiled `.so` is what the solver actually executes —
hashing exactly that makes the equivalence relation provable: same hash
implies same `.so` implies same numerical behaviour.

## Cache hit/miss flow

`getext()` performs a three-tier lookup, cheapest first:

1. **In-memory `_ext_dict`** — same Python process, just a dict access.
2. **On-disk `{hash}.so` + `{hash}.manifest.json`** — same machine, fresh
   process. `load_module` re-loads the `.so` via Python's import machinery
   and verifies the saved constants list still matches the current call's
   manifest (belt-and-braces: a name shift means an unnoticed ABI drift).
3. **Cold compile** via `compile_and_load`. After a successful compile, the
   `.so` and manifest are copied into the cache directory (rank 0 only),
   guarded by an advisory `flock`.

## Invalidation

A cached entry stops being valid when *anything* it implicitly depended on
changes. We handle the common categories like this:

| Change                              | Handled by                                       |
|-------------------------------------|--------------------------------------------------|
| Solver expressions / topology       | `source_hash` differs ⇒ different entry          |
| Constant **value**                  | `source_hash` is independent of value (uses `_JITConstant` placeholders); `PetscDSSetConstants` updates the array at solve time |
| PETSc version, UW version           | Embedded in the ABI salt ⇒ different `source_hash` |
| Compiler / CFLAGS / Python ABI      | `./uw build` writes `.env-fingerprint`; mismatch ⇒ wipe `*.so` and `*.manifest.json` |
| Constant rename                     | `load_module` compares saved vs current names ⇒ miss |
| Manifest schema change              | `MANIFEST_VERSION` bump ⇒ miss                   |

The cache directory is safe to delete by hand at any time:

```
rm -rf ~/.cache/underworld3/jit
```

The next solve will repopulate.

## MPI

When `mpi.size > 1`:

- Every rank computes the C-source hash independently. The hashes are
  `comm.allgather`'d and compared — a mismatch raises immediately rather
  than letting ranks diverge. Non-determinism in `generate_c_source` (e.g.
  set/dict iteration order leaking into emitted C) would land here.
- Rank 0 writes the cache entry; other ranks rely on the disk-cache hit
  path on subsequent calls.
- The `flock` on the per-hash lockfile serialises cross-shell concurrent
  writes (e.g. a `mpirun -np 4` and a `mpirun -np 2` started seconds apart
  on the same machine).

A future refinement would have rank 0 compile while other ranks wait on a
barrier and read the resulting `.so` directly; today every rank still
performs the cold compile but only rank 0 publishes the result.

## Environment variables

| Variable             | Effect                                                  |
|----------------------|---------------------------------------------------------|
| `UW_JIT_CACHE`       | Set to `0`/`false`/`no` to disable disk cache           |
| `UW_JIT_CACHE_DIR`   | Override the cache directory location                   |
| `XDG_CACHE_HOME`     | Used when `UW_JIT_CACHE_DIR` is unset                   |

## Code references

- `src/underworld3/utilities/_jitextension.py` — `getext`, `generate_c_source`,
  `compile_and_load`, `_abi_salt`, `_extract_constants`.
- `src/underworld3/utilities/_jit_cache.py` — disk cache: `get_cache_dir`,
  `load_module`, `store_module`, `_file_lock`.
- `uw` (shell driver) — `.env-fingerprint` write + cache wipe inside
  `run_build`.
