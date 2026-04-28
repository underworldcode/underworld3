from typing import Optional
import os
import shutil
import subprocess
from xmlrpc.client import boolean
import sympy
import underworld3
import underworld3.timing as timing
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path


def _stable_sort_key(obj):
    """Return a process-stable key for ordering symbolic/JIT objects.

    Python hash randomisation means unordered containers such as ``set`` can
    iterate differently on different MPI ranks. JIT source emission must not
    depend on that ordering because every rank has to compile/load byte
    identical C source.
    """

    field_id = getattr(obj, "field_id", None)
    if field_id is None:
        field_id = -1

    return (
        field_id,
        type(obj).__module__,
        type(obj).__qualname__,
        getattr(obj, "name", ""),
        getattr(obj, "clean_name", ""),
        getattr(obj, "_ccodestr", ""),
        str(obj),
    )


def _stable_sorted(iterable):
    """Sort *iterable* with a deterministic key suitable for JIT emission."""

    return sorted(iterable, key=_stable_sort_key)


def _petsc_build_env():
    """Return a subprocess environment with PETSc's C/C++ compilers set.

    Underworld's runtime JIT path shells out to a temporary ``setup.py``
    build. On some platforms the default compiler discovered by setuptools
    is not the same compiler family / wrapper PETSc was built with. Reuse
    PETSc's recorded ``CC`` and ``CXX`` from ``petscvariables`` so the JIT
    build follows the same toolchain as the main package build.
    """

    env = os.environ.copy()

    try:
        import petsc4py

        petsc_info = petsc4py.get_config()
        petsc_dir = petsc_info.get("PETSC_DIR", "")
        petsc_arch = petsc_info.get("PETSC_ARCH", "")
    except Exception:
        return env

    if not petsc_dir:
        return env

    candidate_paths = []
    if petsc_arch:
        candidate_paths.append(
            Path(petsc_dir) / petsc_arch / "lib" / "petsc" / "conf" / "petscvariables"
        )
    candidate_paths.append(Path(petsc_dir) / "lib" / "petsc" / "conf" / "petscvariables")

    petscvars = next((path for path in candidate_paths if path.exists()), None)
    if petscvars is None:
        return env

    cc = ""
    cxx = ""
    with petscvars.open("r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("CC ="):
                cc = line.split("=", 1)[1].strip()
            elif line.startswith("CXX ="):
                cxx = line.split("=", 1)[1].strip()

    if cc:
        env["CC"] = cc
    if cxx:
        env["CXX"] = cxx

    def _openmpi_wrapper_fallback(wrapper, env_key):
        try:
            wrapped = subprocess.check_output(
                [wrapper, "--showme:command"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except Exception:
            return

        if not wrapped:
            return

        compiler = wrapped.split()[0]
        if shutil.which(compiler):
            return

        fallback_name = None
        if "clang++" in compiler:
            fallback_name = "clang++"
        elif "clang" in compiler:
            fallback_name = "clang"
        elif "g++" in compiler or compiler.endswith("c++"):
            fallback_name = "g++"
        elif "gcc" in compiler or compiler.endswith("cc"):
            fallback_name = "cc"

        if not fallback_name:
            return

        fallback = shutil.which(fallback_name)
        if fallback:
            env[env_key] = fallback

    if cc:
        _openmpi_wrapper_fallback(cc, "OMPI_CC")
    if cxx:
        _openmpi_wrapper_fallback(cxx, "OMPI_CXX")

    return env


## This is not required in sympy >= 1.9

# def diff_fn1_wrt_fn2(fn1, fn2):
#     """
#     This function takes the derivative of a function (fn1) with respect
#     to another function (fn2). Sympy does not allow this natively, instead
#     only allowing derivatives with respect to symbols.  Here, we
#     temporarily subsitute fn2 for a dummy symbol, perform the derivative (with
#     respect to the dummy symbol), and then replace the dummy for fn2 again.
#     """
#     if fn2.is_zero:
#         return 0
#     # If fn1 doesn't contain fn2, immediately return zero.
#     # The full diff method will also return zero, but will be slower.
#     if len(fn1.atoms(fn2))==0:
#         return 0
#     uwderivdummy = sympy.Symbol("uwderivdummy")
#     subfn   = fn1.xreplace({fn2:uwderivdummy})      # sub in dummy
#     subfn_d = subfn.diff(uwderivdummy)              # actual deriv
#     deriv   = subfn_d.xreplace({uwderivdummy:fn2})  # sub out dummy
#     return deriv

# In-memory cache of compiled modules, keyed by a hex-string hash of the
# generated C source (plus an ABI salt). Tests count ``len(_ext_dict)`` to
# verify that a parameter-value change does not trigger a new compile.
_ext_dict = {}


def _abi_salt():
    """Return a string that invalidates the cache when binary compatibility changes.

    Keeps the salt conservative (PETSc version + underworld3 version). The
    ``./uw`` build driver is responsible for wiping the on-disk cache when
    the compiler toolchain or Python ABI changes (see design doc).
    """
    try:
        from petsc4py import PETSc
        petsc_ver = ".".join(str(x) for x in PETSc.Sys.getVersion())
    except Exception:
        petsc_ver = "unknown"
    try:
        uw_ver = underworld3.__version__
    except AttributeError:
        uw_ver = "unknown"
    return f"petsc={petsc_ver}|uw={uw_ver}"


# ============================================================================
# JIT Callback Set
# ============================================================================
#
# Groups the five callback lists that PETSc requires for pointwise functions.
# Using a structured container prevents cache-key collisions between callback
# roles (e.g. volume residual vs boundary residual) that share the same
# symbolic form.
# ============================================================================

@dataclass(frozen=True)
class JITCallbackSet:
    """Immutable container for the five PETSc pointwise callback lists.

    Each slot holds a tuple of SymPy expressions for one callback role.
    The structured representation ensures that cache keys preserve which
    role each expression belongs to, preventing the collision bug where
    ``Integral(fn=1)`` and ``BdIntegral(fn=1)`` would share a cached module.

    Parameters
    ----------
    residual : tuple
        Volume residual expressions (F0, F1 for each field).
    bcs : tuple
        Essential boundary condition expressions.
    jacobian : tuple
        Jacobian expressions (G0, G1, G2, G3 for each field pair).
    bd_residual : tuple
        Boundary residual expressions (includes ``petsc_n[]`` access).
    bd_jacobian : tuple
        Boundary Jacobian expressions.
    """
    residual: tuple = ()
    bcs: tuple = ()
    jacobian: tuple = ()
    bd_residual: tuple = ()
    bd_jacobian: tuple = ()

    def __post_init__(self):
        """Coerce all slots to tuples for immutability and hashability."""
        for field in ("residual", "bcs", "jacobian", "bd_residual", "bd_jacobian"):
            val = getattr(self, field)
            if val is None:
                object.__setattr__(self, field, ())
            elif not isinstance(val, tuple):
                object.__setattr__(self, field, tuple(val))

    def flat(self) -> tuple:
        """Concatenate all slots into a single ordered tuple.

        The ordering (residual, bcs, jacobian, bd_residual, bd_jacobian)
        matches what ``_createext()`` expects.
        """
        return self.residual + self.bcs + self.jacobian + self.bd_residual + self.bd_jacobian

    def signature(self) -> tuple:
        """Hashable key that preserves callback role separation.

        Two callback sets with the same expressions in different roles
        will produce different signatures.
        """
        return (self.residual, self.bcs, self.jacobian, self.bd_residual, self.bd_jacobian)

    def map(self, fn) -> 'JITCallbackSet':
        """Apply *fn* to every expression in every slot, returning a new set."""
        return JITCallbackSet(
            residual=tuple(fn(f) for f in self.residual),
            bcs=tuple(fn(f) for f in self.bcs),
            jacobian=tuple(fn(f) for f in self.jacobian),
            bd_residual=tuple(fn(f) for f in self.bd_residual),
            bd_jacobian=tuple(fn(f) for f in self.bd_jacobian),
        )

    @property
    def counts(self):
        """Lengths of each slot, for ``_createext()`` offset calculation."""
        return (len(self.residual), len(self.bcs), len(self.jacobian),
                len(self.bd_residual), len(self.bd_jacobian))


def prepare_for_cache_key(fn, constants_subs_map):
    """Prepare a single expression for JIT cache hashing.

    Two-phase process:
    1. Substitute constant UWexpressions with ``_JITConstant`` placeholders
       so that changing a constant's *value* does not invalidate the cache.
    2. Unwrap remaining (non-constant) UWexpressions to pure SymPy so the
       hash is deterministic.

    Parameters
    ----------
    fn : sympy expression or None
        The expression to expand.
    constants_subs_map : dict or None
        Mapping from UWexpression symbols to ``_JITConstant`` placeholders.
    """
    # Phase 1: Substitute constants with _JITConstant placeholders
    if constants_subs_map and fn is not None:
        try:
            fn_structural = fn.xreplace(constants_subs_map) if hasattr(fn, "xreplace") else fn
        except Exception:
            fn_structural = fn
    else:
        fn_structural = fn

    # Phase 2: Unwrap remaining (non-constant) expressions
    return underworld3.function.expressions.unwrap(
        fn_structural, keep_constants=False, return_self=False
    )


# ============================================================================
# JIT Constants Support
# ============================================================================
#
# UWexpressions that are "constant" (no spatial/field dependencies) are routed
# through PETSc's constants[] array instead of being baked as C literals.
# This allows parameter changes without JIT recompilation.
# ============================================================================

class _JITConstant(sympy.Symbol):
    """Symbol subclass that renders as constants[i] in generated C code.

    Used by the JIT compiler to route constant UWexpressions through
    PETSc's PetscDSSetConstants() mechanism instead of baking values
    as C literals.
    """

    def __new__(cls, index, name=None):
        if name is None:
            name = f"_jit_const_{index}"
        obj = super().__new__(cls, name)
        obj._const_index = index
        obj._ccodestr = f"constants[{index}]"
        return obj

    def _ccode(self, printer):
        return self._ccodestr


def _extract_constants(all_fns, mesh):
    """Extract constant UWexpressions from a list of pre-unwrap functions.

    Scans all expressions for UWexpression atoms where is_constant_expr()
    is True (no spatial/field dependencies). Assigns deterministic indices
    sorted by expression name for MPI consistency.

    Parameters
    ----------
    all_fns : tuple of sympy expressions
        The raw (pre-unwrap) function list.
    mesh : underworld3.discretisation.Mesh
        The mesh (currently unused, reserved for future mesh.t support).

    Returns
    -------
    list of (int, UWexpression)
        Ordered mapping from constants[] index to UWexpression reference.
    dict
        Mapping from UWexpression to _JITConstant symbol for substitution.
    """
    from underworld3.function.expressions import (
        is_constant_expr,
        extract_expressions,
        UWexpression,
    )

    constant_exprs = set()

    for fn in all_fns:
        if fn is None:
            continue

        # Handle Matrix expressions
        if isinstance(fn, sympy.MatrixBase):
            for elem in fn:
                _collect_constant_atoms(elem, constant_exprs, is_constant_expr, UWexpression)
        else:
            _collect_constant_atoms(fn, constant_exprs, is_constant_expr, UWexpression)

    if not constant_exprs:
        return [], {}

    # Sort by the user-given symbol name, not ``str(expr)`` — ``__str__`` on a
    # UWexpression returns the current *value*, which shuffles the index
    # assignment whenever a value changes. ``.name`` is stable.
    sorted_constants = sorted(constant_exprs, key=lambda e: (e.name, _stable_sort_key(e)))

    manifest = []
    subs_map = {}
    for i, expr in enumerate(sorted_constants):
        # Use ``expr.name`` (stable) instead of ``str(expr)`` (= current value)
        # so the placeholder symbol's identity is independent of parameter value.
        jit_const = _JITConstant(i, name=f"_jit_const_{expr.name}")
        manifest.append((i, expr))
        subs_map[expr] = jit_const

    return manifest, subs_map


def _is_truly_constant(expr, UWexpression):
    """Check if a UWexpression resolves to a pure constant (no spatial deps).

    Unlike is_constant_expr(), this handles nested UWexpressions correctly
    by fully unwrapping and checking if the result has any spatial/field
    symbols (BaseScalar, UnderworldFunction, etc.).
    """
    try:
        unwrapped = underworld3.function.expressions.unwrap_expression(
            expr, mode='nondimensional'
        )
    except Exception:
        return False

    # If it unwraps to a plain number, it's constant
    if isinstance(unwrapped, (int, float)):
        return True
    if isinstance(unwrapped, sympy.Number):
        return True

    if not hasattr(unwrapped, 'free_symbols'):
        try:
            float(unwrapped)
            return True
        except (TypeError, ValueError):
            return False

    # Check remaining free symbols — any spatial/field dependency makes it non-constant
    from sympy.vector.scalar import BaseScalar
    for sym in _stable_sorted(unwrapped.free_symbols):
        if isinstance(sym, BaseScalar):
            return False
        if isinstance(sym, sympy.Function):
            return False
        # UnderworldFunction symbols have _ccodestr pointing to petsc arrays
        if hasattr(sym, '_ccodestr') and not isinstance(sym, _JITConstant):
            ccode = sym._ccodestr
            if 'petsc_u' in ccode or 'petsc_a' in ccode or 'petsc_x' in ccode or 'petsc_n' in ccode:
                return False
        # Other UWexpressions that didn't fully unwrap — not constant
        if isinstance(sym, UWexpression):
            return False

    return True


def _collect_constant_atoms(expr, result_set, is_constant_expr, UWexpression):
    """Recursively collect constant UWexpression atoms from an expression."""

    if isinstance(expr, UWexpression):
        if _is_truly_constant(expr, UWexpression):
            result_set.add(expr)
            return  # Don't recurse into constant expressions
        # Non-constant UWexpression: check its inner sym for nested constants
        if hasattr(expr, '_sym') and expr._sym is not None:
            _collect_constant_atoms(expr._sym, result_set, is_constant_expr, UWexpression)
        return

    if not hasattr(expr, 'atoms'):
        return

    # Check all UWexpression atoms
    for atom in _stable_sorted(expr.atoms(sympy.Symbol)):
        if isinstance(atom, UWexpression) and _is_truly_constant(atom, UWexpression):
            result_set.add(atom)
        elif isinstance(atom, UWexpression):
            # Non-constant UWexpression: recurse into its sym
            if hasattr(atom, '_sym') and atom._sym is not None:
                _collect_constant_atoms(atom._sym, result_set, is_constant_expr, UWexpression)


def _pack_constants(manifest):
    """Pack current values from a constants manifest into a flat array.

    Parameters
    ----------
    manifest : list of (int, UWexpression)
        The constants manifest from _extract_constants().

    Returns
    -------
    list of float
        Current nondimensional values in index order.
    """
    import numpy as np

    if not manifest:
        return np.array([], dtype=np.float64)

    values = np.zeros(len(manifest), dtype=np.float64)
    for idx, uw_expr in manifest:
        try:
            values[idx] = float(
                underworld3.function.expressions.unwrap_expression(
                    uw_expr, mode='nondimensional'
                )
            )
        except (TypeError, ValueError):
            # Fallback: try .data property
            try:
                values[idx] = float(uw_expr.data)
            except Exception:
                values[idx] = 0.0
    return values


# Generates the C debugging string for the compiled function block
def debugging_text(randstr, fn, fn_type, eqn_no):
    try:
        object_size = len(fn.flat())
    except:
        object_size = 1

    outstr = "out[0]"
    for i in range(1, object_size):
        outstr += f", out[{i}]"

    formatstr = "%6e, " * object_size

    debug_str = f"/* {fn} */\n"
    debug_str += f"/* Size = {object_size} */\n"
    debug_str += f'FILE *fp; fp = fopen( "{randstr}_debug.txt", "a" );\n'
    debug_str += f'fprintf(fp,"{fn_type} - equation {eqn_no} at (%.2e, %.2e, %.2e) -> ", petsc_x[0], petsc_x[1], dim==2 ? 0.0: petsc_x[2]);\n'
    debug_str += f'fprintf(fp,"{formatstr}\\n", {outstr});\n'
    debug_str += f"fclose(fp);"

    return debug_str


def debugging_text_bd(randstr, fn, fn_type, eqn_no):
    try:
        object_size = len(fn.flat())
    except:
        object_size = 1

    outstr = "out[0]"
    for i in range(1, object_size):
        outstr += f", out[{i}]"

    formatstr = "%6e, " * object_size

    debug_str = f"/* {fn} */\n"
    debug_str += f"/* Size = {object_size} */\n"
    debug_str += f'FILE *fp; fp = fopen( "{randstr}_debug.txt", "a" );\n'
    debug_str += f'fprintf(fp,"{fn_type} - equation {eqn_no} X / N (%.2e, %.2e, %.2e / %2.e, %2.e, %.2e ) -> ", petsc_x[0], petsc_x[1], dim==2 ? 0.0: petsc_x[2], petsc_n[0], petsc_n[1], dim==2 ? 0.0: petsc_n[2]);\n'
    debug_str += f'fprintf(fp,"{formatstr}\\n", {outstr});\n'
    debug_str += f"fclose(fp);"

    return debug_str


_GextResult = namedtuple("GextResult", ["ptrobj", "fn_dicts", "constants_manifest", "cache_key"])


@timing.routine_timer_decorator
def getext(
    mesh,
    callbacks: JITCallbackSet,
    primary_field_list,
    verbose=False,
    debug=False,
    debug_name=None,
    cache=True,
):
    """Compile (or retrieve cached) JIT extension for PETSc pointwise functions.

    Parameters
    ----------
    mesh : Mesh
        Supporting mesh for coordinate system and variable information.
    callbacks : JITCallbackSet
        Callback expressions grouped by PETSc role (residual, bcs, jacobian,
        bd_residual, bd_jacobian).
    primary_field_list : iterable
        Variables that map to PETSc primary arrays (``petsc_u[]``).
        All others map to auxiliary arrays (``petsc_a[]``).

    Returns
    -------
    GextResult
        Named tuple with fields (ptrobj, fn_dicts, constants_manifest).
        constants_manifest is a list of (index, uw_expression_ref) tuples
        for use with PetscDSSetConstants().
    """
    import hashlib

    primary_field_list = tuple(primary_field_list)

    # Extract constant UWexpressions that are routed through PETSc's
    # constants[] array. Value changes don't affect the C source — they
    # only alter what we pass to PetscDSSetConstants at solve time.
    constants_manifest, constants_subs_map = _extract_constants(callbacks.flat(), mesh)

    if debug and underworld3.mpi.rank == 0:
        if constants_manifest:
            print(f"Constants manifest ({len(constants_manifest)} entries):")
            for idx, expr in constants_manifest:
                print(f"  constants[{idx}] = {expr.name} (current value: {expr.data})")

    # Generate C source. The returned ``gen_modname``/``gen_randstr`` are
    # implementation-dependent (often random) — we canonicalise them below
    # so byte-identical inputs hash to the same key.
    _PLACEHOLDER_NAME = "UWJITPLACEHOLDER"
    gen_modname, codeguys, diag = generate_c_source(
        _PLACEHOLDER_NAME,
        mesh,
        callbacks,
        primary_field_list,
        constants_subs_map=constants_subs_map,
        verbose=verbose,
        debug=debug,
        debug_name=debug_name,
    )
    gen_randstr = diag["randstr"]

    # Canonicalise: swap the call-specific identifiers for stable tokens so
    # the hash is a function of *what the module does*, not what it happens
    # to be named. The same trick lets us derive a unique-per-bundle, yet
    # deterministic, C-symbol prefix below (required by some dlopen loaders
    # that use RTLD_GLOBAL — see historical comment on ``randstr``).
    _CAN_MOD = "__UW_JIT_MOD__"
    _CAN_RS = "__UW_JIT_RS__"
    canonical_codeguys = [
        [
            entry[0],
            entry[1].replace(gen_modname, _CAN_MOD).replace(gen_randstr, _CAN_RS),
        ]
        for entry in codeguys
    ]
    canonical_source = "\n".join(entry[1] for entry in canonical_codeguys)
    source_hash = hashlib.sha256(
        (canonical_source + "\n---\n" + _abi_salt()).encode("utf-8")
    ).hexdigest()[:16]

    # Determinism check: all ranks must agree on the hash. A mismatch means
    # generate_c_source isn't deterministic across ranks (typically caused
    # by set/dict-iteration order leaking into the emitted C). Caching
    # cannot work correctly if ranks disagree, so fail loudly rather than
    # let stale entries propagate.
    if underworld3.mpi.size > 1:
        all_hashes = underworld3.mpi.comm.allgather(source_hash)
        if any(h != source_hash for h in all_hashes):
            raise RuntimeError(
                f"JIT C-source hash differs across MPI ranks: {set(all_hashes)}. "
                f"This indicates non-determinism in generate_c_source — likely "
                f"a set or dict whose iteration order leaks into the C output. "
                f"Treating this as a hard error since cache reuse would be unsound."
            )

    # Derive the real modname/randstr from the hash — same source ⇒ same
    # compiled artefact, different sources ⇒ disjoint symbol namespaces.
    real_modname = f"fn_ptr_ext_{source_hash}"
    real_randstr = "UW" + source_hash[:8].upper()
    codeguys_final = [
        [
            entry[0],
            entry[1].replace(_CAN_MOD, real_modname).replace(_CAN_RS, real_randstr),
        ]
        for entry in canonical_codeguys
    ]

    # ── Cache lookup or build ─────────────────────────────────────────────
    # Three tiers, cheapest first:
    #   (1) in-memory dict — same Python process
    #   (2) on-disk cache — same machine, different process
    #   (3) cold compile via Cython + cc (rank 0 only when MPI > 1)
    if cache and source_hash in _ext_dict:
        if verbose and underworld3.mpi.rank == 0:
            print(f"JIT compiled module cached (memory) ... {source_hash}", flush=True)
        module = _ext_dict[source_hash]
    else:
        from underworld3.utilities import _jit_cache as _jc

        # Disk lookup is cheap and rank-local — every rank checks
        # independently. UW assumes a shared filesystem for the cache dir.
        module = None
        if cache:
            module = _jc.load_module(source_hash, real_modname, constants_manifest)
            if module is not None and verbose and underworld3.mpi.rank == 0:
                print(f"JIT compiled module cached (disk) ... {source_hash}", flush=True)

        if module is None:
            # Cold compile path. With MPI, only rank 0 invokes cc and
            # publishes; the other ranks barrier-wait and then load the
            # freshly-published .so from disk. This avoids the N× cc
            # storm on cold-start under mpirun -np N.
            #
            # Falls back to "every rank compiles" only when the disk
            # cache is disabled (no way to share a build), in which
            # case the wasted cc cost is on the user's chosen path.
            multi_rank = underworld3.mpi.size > 1
            disk_enabled = cache and _jc.get_cache_dir() is not None

            if multi_rank and disk_enabled:
                if underworld3.mpi.rank == 0:
                    if verbose:
                        print(
                            f"JIT compiling new module on rank 0 ... {source_hash}",
                            flush=True,
                        )
                    module, tmpdir = compile_and_load(
                        real_modname, codeguys_final, verbose=verbose
                    )
                    if verbose:
                        print(f"Location of compiled module: {tmpdir}", flush=True)
                    _jc.store_module(
                        source_hash, real_modname, tmpdir, constants_manifest
                    )
                # Synchronisation point: rank 0 has now published the
                # entry; other ranks may load it.
                underworld3.mpi.comm.Barrier()
                if underworld3.mpi.rank != 0:
                    module = _jc.load_module(
                        source_hash, real_modname, constants_manifest
                    )
                    if module is None:
                        # Defensive: rank-0 publish failed (e.g. write
                        # error). Fall back to a local compile so this
                        # rank is at least correct, even if wasteful.
                        module, _tmpdir = compile_and_load(
                            real_modname, codeguys_final, verbose=verbose
                        )
            else:
                if verbose and underworld3.mpi.rank == 0:
                    print(
                        f"JIT compiling new module ... {source_hash}", flush=True
                    )
                module, tmpdir = compile_and_load(
                    real_modname, codeguys_final, verbose=verbose
                )
                if verbose and underworld3.mpi.rank == 0:
                    # Tests in test_0004_pointwise_fns parse this exact prefix
                    # to find the per-call build directory.
                    print(f"Location of compiled module: {tmpdir}", flush=True)
                if cache:
                    _jc.store_module(
                        source_hash, real_modname, tmpdir, constants_manifest
                    )

        if cache:
            _ext_dict[source_hash] = module

    ptrobj = module.getptrobj()

    # Build the slot-index dicts. Keys are the original callback objects so
    # solvers can do ``ext.fns_residual[ext_dict.res[self._u_f0]]``.
    i_res = {fn: i for i, fn in enumerate(callbacks.residual)}
    i_ebc = {fn: i for i, fn in enumerate(callbacks.bcs)}
    i_jac = {fn: i for i, fn in enumerate(callbacks.jacobian)}
    i_bd_res = {fn: i for i, fn in enumerate(callbacks.bd_residual)}
    i_bd_jac = {fn: i for i, fn in enumerate(callbacks.bd_jacobian)}

    extn_fn_dict = namedtuple(
        "Functions", ["res", "jac", "ebc", "bd_res", "bd_jac"],
    )

    return _GextResult(
        ptrobj,
        extn_fn_dict(i_res, i_jac, i_ebc, i_bd_res, i_bd_jac),
        constants_manifest,
        cache_key=source_hash,
    )


@timing.routine_timer_decorator
def generate_c_source(
    name,
    mesh: underworld3.discretisation.Mesh,
    callbacks: JITCallbackSet,
    primary_field_list,
    constants_subs_map: Optional[dict] = None,
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    debug_name=None,
):
    """Generate the setup.py / C header / Cython wrapper for a JIT bundle.

    This is the pure text-generation phase: sympy processing, C-code emission,
    and assembly of the files that will make up the compiled module. No I/O,
    no subprocess, no dynamic loading — those happen in ``compile_and_load``.

    Keying a cache on a hash of the generated C source requires that this
    function produce byte-identical output for byte-identical inputs.

    Parameters
    ----------
    name : str or int
        Identifier used to build ``MODNAME = "fn_ptr_ext_" + str(name)``.
    mesh : Mesh
    callbacks : JITCallbackSet
    primary_field_list : list
        Variables that map to PETSc primary variable arrays (``petsc_u[]``).
    constants_subs_map : dict, optional
        Mapping from UWexpression → ``_JITConstant`` placeholder.

    Returns
    -------
    modname : str
        Fully-qualified extension module name (``fn_ptr_ext_<name>``).
    codeguys : list of [filename, content]
        The files that make up the source bundle
        (``setup.py``, ``cy_ext.h``, ``cy_ext.pyx``).
    diagnostics : dict
        Equation-range counts and the random symbol prefix, used by the caller
        for verbose printing and for building the fn-layout manifest.
    """
    from sympy import symbols, Eq, MatrixSymbol
    from underworld3 import VarType

    fns = callbacks.flat()
    count_residual_sig, count_bc_sig, count_jacobian_sig, \
        count_bd_residual_sig, count_bd_jacobian_sig = callbacks.counts

    # `_ccode` patching
    def ccode_patch_fns(varlist, prefix_str):
        """
        This function patches uw functions with the necessary ccode
        routines for the code printing.

        For a `varlist` consisting of 2d velocity & pressure variables,
        for example, it'll generate routines which write the following,
        where `prefix_str="petsc_u"`:
            V_x   : "petsc_u[0]"
            V_y   : "petsc_u[1]"
            P     : "petsc_u[2]"
            V_x_x : "petsc_u_x[0]"
            V_x_y : "petsc_u_x[1]"
            V_y_x : "petsc_u_x[2]"
            V_y_y : "petsc_u_x[3]"
            P_x   : "petsc_u_x[4]"
            P_y   : "petsc_u_x[5]"

        Params
        ------
        varlist: list
            The variables to patch. Note that *all* the variables in the
            corresponding `PetscDM` must be included. They must also be
            ordered according to their `field_id`.
        prefix_str: str
            The string prefix to write.
        """
        u_i = 0  # variable increment
        u_x_i = 0  # variable gradient increment
        lambdafunc = lambda self, printer: self._ccodestr
        for var in varlist:
            if var.vtype == VarType.SCALAR:
                # monkey patch this guy into the function
                type(var.fn)._ccodestr = f"{prefix_str}[{u_i}]"
                type(var.fn)._ccode = lambdafunc
                u_i += 1
                # now patch gradient guy into varfn guy
                for ind in range(mesh.dim):
                    # Note that var.fn._diff[ind] returns the class, so we don't need type(var.fn._diff[ind])
                    var.fn._diff[ind]._ccodestr = f"{prefix_str}_x[{u_x_i}]"
                    var.fn._diff[ind]._ccode = lambdafunc
                    u_x_i += 1
            elif (
                var.vtype == VarType.VECTOR
                or var.vtype == VarType.TENSOR
                or var.vtype == VarType.SYM_TENSOR
                or var.vtype == VarType.MATRIX
            ):
                # Pull out individual sub components
                for comp in var.sym_1d:
                    # monkey patch
                    type(comp)._ccodestr = f"{prefix_str}[{u_i}]"
                    type(comp)._ccode = lambdafunc
                    u_i += 1
                    # and also patch gradient guy into varfn guy's comp guy   # Argh ... too much Mansourness
                    for ind in range(mesh.dim):
                        # Note that var.fn._diff[ind] returns the class, so we don't need type(var.fn._diff[ind])
                        comp._diff[ind]._ccodestr = f"{prefix_str}_x[{u_x_i}]"
                        comp._diff[ind]._ccode = lambdafunc
                        u_x_i += 1
            else:
                raise RuntimeError(
                    f"Unsupported type {var.vtype} for code generation. Please contact developers."
                )

    # Patch in `_code` methods. Note that the order here
    # is important, as the secondary call will overwrite
    # those patched in the first call.

    ccode_patch_fns(_stable_sorted(mesh.vars.values()), "petsc_a")
    ccode_patch_fns(primary_field_list, "petsc_u")

    # Also patch `BaseScalar` types. Nothing fancy - patch the overall type,
    # make sure each component points to the correct PETSc data

    ## This is set up in the mesh at the moment but this does seem to be the wrong place

    # mesh.N.x._ccodestr = "petsc_x[0]"
    # mesh.N.y._ccodestr = "petsc_x[1]"
    # mesh.N.z._ccodestr = "petsc_x[2]"

    # # Surface integrals also have normal vector information as petsc_n

    # mesh.Gamma_N.x._ccodestr = "petsc_n[0]"
    # mesh.Gamma_N.y._ccodestr = "petsc_n[1]"
    # mesh.Gamma_N.z._ccodestr = "petsc_n[2]"

    def _basescalar_ccode(self, printer):
        """C code for coordinate symbols, with fallback for new instances.

        sympy.simplify() may create new BaseScalar/UWCoordinate instances
        that lack _ccodestr. We recover it from the coordinate's _id attribute
        which stores (index, system_name).
        """
        if hasattr(self, '_ccodestr'):
            return self._ccodestr
        # Fallback: compute from _id
        idx = self._id[0]
        system_name = str(self._id[1])
        if 'Gamma' in system_name:
            return f"petsc_n[{idx}]"
        else:
            return f"petsc_x[{idx}]"

    type(mesh.N.x)._ccode = _basescalar_ccode
    # Gamma base scalars (un-normalised face normal) — ensure ccode is registered
    Gamma_scalars = mesh._Gamma.base_scalars()
    if type(Gamma_scalars[0]) is not type(mesh.N.x):
        type(Gamma_scalars[0])._ccode = _basescalar_ccode

    # Create a custom functions replacement dictionary.
    # Note that this dictionary is really just to appease Sympy,
    # and the actual implementation is printed directly into the
    # generated JIT files (see `h_str` below). Without specifying
    # this dictionary, Sympy doesn't code print the Heaviside correctly.
    # For example, it will print
    #    Heaviside(petsc_x[0,1])
    # instead of
    #    Heaviside(petsc_x[1]).
    # Note that the Heaviside implementation will be printed into all JIT
    # files now. This is fine for now, but if more complex functions are
    # required a cleaner solution might be desirable.

    custom_functions = {
        "Heaviside": [
            (
                lambda *args: len(args) == 1,
                "Heaviside_1",
            ),  # for single arg Heaviside  (defaults to 0.5 at jump).
            (lambda *args: len(args) == 2, "Heaviside_2"),
        ],  # for two arg Heavisides    (second arg is jump value).
    }

    # Now go ahead and generate C code from substituted Sympy expressions.
    # from sympy.printing.c import C99CodePrinter
    # printer = C99CodePrinter(user_functions=custom_functions)
    from sympy.printing.c import c_code_printers

    printer = c_code_printers["c99"]({"user_functions": custom_functions})

    # Purge libary/header dictionaries. These will be repopulated
    # when `doprint` is called below. This ensures that we only link
    # in libraries where needed.
    # Note that this generally shouldn't be necessary, as the
    # extension module should build successfully even where
    # libraries are linked in redundantly. However it does
    # help to ensure that any potential linking issues are isolated
    # to only those sympy functions (just analytic solutions currently)
    # that require linking. There may also be a performance advantage
    # (faster extension build time) but this is unlikely to be
    # significant.
    underworld3._incdirs.clear()
    underworld3._libdirs.clear()
    underworld3._libfiles.clear()

    eqns = []
    for index, fn in enumerate(fns):

        # Save original for debugging
        fn_original = fn

        # Two-phase unwrap:
        # Phase 1: Substitute constant UWexpressions with _JITConstant symbols
        #          These survive into C code as constants[i]
        if constants_subs_map and fn is not None:
            try:
                fn = fn.xreplace(constants_subs_map) if hasattr(fn, 'xreplace') else fn
            except Exception:
                pass

        # Phase 2: Unwrap remaining non-constant UWexpressions to numerical values
        fn = underworld3.function.expressions.unwrap(fn, keep_constants=False, return_self=False)

        if isinstance(fn, sympy.vector.Vector):
            fn = fn.to_matrix(mesh.N)[0 : mesh.dim, 0]
        elif isinstance(fn, sympy.vector.Dyadic):
            fn = fn.to_matrix(mesh.N)[0 : mesh.dim, 0 : mesh.dim]
        else:
            fn = sympy.Matrix([fn])

        # === COORDINATE SYMBOL RECOVERY ===
        # When sympy.simplify() manipulates expressions containing coordinate
        # symbols (BaseScalar/UWCoordinate), it may create NEW instances that
        # lack the _ccodestr attribute set during mesh initialization.
        # This commonly occurs with coordinate-dependent constitutive models
        # (e.g., TransverseIsotropicFlowModel with a radial director).
        # We recover _ccodestr from the coordinate's _id attribute.
        from sympy.vector.scalar import BaseScalar

        free_syms = tuple(_stable_sorted(fn.free_symbols))
        for sym in free_syms:
            if isinstance(sym, BaseScalar) and not hasattr(sym, '_ccodestr'):
                idx = sym._id[0]  # 0, 1, or 2 for x, y, z
                system_name = str(sym._id[1])
                if 'Gamma' in system_name:
                    sym._ccodestr = f"petsc_n[{idx}]"
                else:
                    sym._ccodestr = f"petsc_x[{idx}]"

        # === JIT VALIDATION GATEWAY ===
        # Check for symbols that cannot be converted to C code.
        # Expected symbols (coordinates) have _ccodestr attribute set.
        # Unexpected symbols indicate malformed expressions from user code.
        unconvertible_symbols = []
        for sym in free_syms:
            # Check if this symbol can be converted to C code
            if not hasattr(sym, '_ccodestr'):
                unconvertible_symbols.append(sym)

        if unconvertible_symbols:
            # Build a helpful error message
            sym_details = []
            for sym in unconvertible_symbols:
                detail = f"  - {sym} (type: {type(sym).__name__})"
                if hasattr(sym, 'units'):
                    detail += f" [has units: {sym.units}]"
                if hasattr(sym, 'value'):
                    detail += f" [value: {sym.value}]"
                sym_details.append(detail)

            raise RuntimeError(
                f"\n{'='*70}\n"
                f"JIT COMPILATION ERROR: Expression contains unconvertible symbols\n"
                f"{'='*70}\n\n"
                f"The following symbols could not be converted to C code:\n"
                + "\n".join(sym_details) + "\n\n"
                f"This usually means:\n"
                f"  1. A UWexpression or UWQuantity was not properly expanded\n"
                f"  2. An arithmetic operation failed (e.g., Matrix * UWexpression)\n"
                f"  3. A symbolic function is missing from the expression tree\n\n"
                f"Expression index: {index}\n"
                f"Original expression: {fn_original}\n"
                f"After unwrap: {fn}\n\n"
                f"TIP: Check that all expression operations (*, /, +, -) produce\n"
                f"valid SymPy expressions. For example, ensure scalar * Matrix\n"
                f"and not Matrix * scalar when using UWexpression objects.\n"
                f"{'='*70}"
            )

        if verbose:
            print("Processing JIT {:4d} / {}".format(index, fn))
            # Enhanced debugging output for remaining (valid) free symbols
            if free_syms:
                print("  Free symbols (all convertible):")
                for sym in free_syms:
                    print(f"    - {sym} (type: {type(sym).__name__}, _ccodestr: {getattr(sym, '_ccodestr', 'N/A')})")

        out = sympy.MatrixSymbol("out", *fn.shape)
        eqn = ("eqn_" + str(index), printer.doprint(fn, out))
        if eqn[1].startswith("// Not supported in C:"):
            spliteqn = eqn[1].split("\n")
            raise RuntimeError(
                f"Error encountered generating JIT extension:\n"
                f"{spliteqn[0]}\n"
                f"{spliteqn[1]}\n"
                f"This is usually because code generation for a Sympy function (or its derivative) is not supported.\n"
                f"Please contact the developers."
                f"---"
                f"The ID of the JIT component that failed is {index}"
                f"The decription of the JIT component that failed:\n {fn}"
            )
        eqns.append(eqn)

    MODNAME = "fn_ptr_ext_" + str(name)

    codeguys = []
    # Create a `setup.py`
    setup_py_str = """
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize

ext_mods = [Extension(
    '{NAME}', ['cy_ext.pyx',],
    include_dirs={HEADERS},
    library_dirs={LIBDIRS},
    runtime_library_dirs={LIBDIRS},
    libraries={LIBFILES},
    extra_compile_args=['-std=c99','-O3'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
""".format(
        NAME=MODNAME,
        HEADERS=list(_stable_sorted(underworld3._incdirs.keys())),
        LIBDIRS=list(_stable_sorted(underworld3._libdirs.keys())),
        LIBFILES=list(_stable_sorted(underworld3._libfiles.keys())),
    )
    codeguys.append(["setup.py", setup_py_str])

    residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    bd_residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], const PetscReal petsc_n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    bd_jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[],  const PetscReal petsc_n[],PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"

    # Create header top content.
    h_str = """
typedef int PetscInt;
typedef double PetscReal;
typedef double PetscScalar;
typedef int PetscBool;
#include <math.h>

// Adding missing function implementation
static inline double Heaviside_1 (double x)                 { return x < 0 ? 0 : x > 0 ? 1 : 0.5;     };
static inline double Heaviside_2 (double x, double mid_val) { return x < 0 ? 0 : x > 0 ? 1 : mid_val; };

"""

    # Create cython top content.
    pyx_str = """
from underworld3.cython.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, PetscDSResidualFn, PetscDSJacobianFn, PetscDSBdResidualFn, PetscDSBdJacobianFn
from underworld3.cython.petsc_types cimport PtrContainer
from libc.stdlib cimport malloc
from libc.math cimport *

cdef extern from "cy_ext.h" nogil:
"""

    # Generate a random string to prepend to symbol names.
    # This is generally not required, but on some systems (depending
    # on how Python is configured to dynamically load libraries)
    # it avoids difficulties with symbol namespace clashing which
    # results in only the first JIT module working (with all
    # subsequent modules pointing towards the first's symbols).
    # Tags: RTLD_LOCAL, RTLD_Global, Gadi.

    import string
    import random
    import os

    if not "UW_JITNAME" in os.environ:
        randstr = "".join(random.choices(string.ascii_uppercase, k=5))
    else:
        if debug_name is None:
            randstr = "FUNC_" + str(len(_ext_dict.keys()))
        else:
            randstr = debug_name

    # Print includes
    for header in _stable_sorted(printer.headers):
        h_str += '#include "{}"\n'.format(header)

    h_str += "\n"

    # Print equations
    eqn_index_0 = 0
    eqn_index_1 = count_residual_sig
    fn_counter = 0

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  res", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], residual_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bc_sig

    # The bcs have the same signature as the residuals (at present)
    # but we leave this separate in case it changes in later PETSc implementations

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  ebc", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], residual_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_jacobian_sig

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  jac", fn_counter)

        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], jacobian_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], jacobian_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bd_residual_sig
    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text_bd(randstr, fns[fn_counter], "bdres", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], bd_residual_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], bd_residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bd_jacobian_sig
    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text_bd(randstr, fns[fn_counter], "bdjac", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], bd_jacobian_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], bd_jacobian_sig)
        fn_counter += 1

    codeguys.append(["cy_ext.h", h_str])
    # Note that the malloc below will cause a leak, but it's just a bunch of function
    # pointers so we don't need to worry about it (yet)
    pyx_str += """
cpdef PtrContainer getptrobj():
    clsguy = PtrContainer()
    clsguy.fns_residual = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))
    clsguy.fns_bcs      = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))
    clsguy.fns_jacobian = <PetscDSJacobianFn*> malloc({}*sizeof(PetscDSJacobianFn))
    clsguy.fns_bd_residual = <PetscDSBdResidualFn*> malloc({}*sizeof(PetscDSBdResidualFn))
    clsguy.fns_bd_jacobian = <PetscDSBdJacobianFn*> malloc({}*sizeof(PetscDSBdJacobianFn))
""".format(
        count_residual_sig,
        count_bc_sig,
        count_jacobian_sig,
        count_bd_residual_sig,
        count_bd_jacobian_sig,
    )

    eqn_count = 0
    for index, eqn in enumerate(eqns[eqn_count : eqn_count + count_residual_sig]):
        pyx_str += "    clsguy.fns_residual[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    residual_equations = (0, eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + count_bc_sig]):
        pyx_str += "    clsguy.fns_bcs[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    boundary_equations = (residual_equations[1], eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + count_jacobian_sig]):
        pyx_str += "    clsguy.fns_jacobian[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    jacobian_equations = (boundary_equations[1], eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + count_bd_residual_sig]):
        pyx_str += "    clsguy.fns_bd_residual[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    boundary_residual_equations = (jacobian_equations[1], eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + count_bd_jacobian_sig]):
        pyx_str += "    clsguy.fns_bd_jacobian[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    boundary_jacobian_equations = (boundary_residual_equations[1], eqn_count)

    pyx_str += "    return clsguy"
    codeguys.append(["cy_ext.pyx", pyx_str])

    diagnostics = {
        "randstr": randstr,
        "eqn_count": eqn_count,
        "count_residual_sig": count_residual_sig,
        "count_bc_sig": count_bc_sig,
        "count_jacobian_sig": count_jacobian_sig,
        "count_bd_residual_sig": count_bd_residual_sig,
        "count_bd_jacobian_sig": count_bd_jacobian_sig,
        "residual_equations": residual_equations,
        "boundary_equations": boundary_equations,
        "jacobian_equations": jacobian_equations,
        "boundary_residual_equations": boundary_residual_equations,
        "boundary_jacobian_equations": boundary_jacobian_equations,
    }
    return MODNAME, codeguys, diagnostics


def compile_and_load(modname, codeguys, verbose=False):
    """Write ``codeguys`` files to a temp directory, build with Cython, and dynamically load.

    Split out of the former ``_createext`` so that generation and compilation
    can be hashed/cached independently.

    Parameters
    ----------
    modname : str
        Fully-qualified name of the extension module (``fn_ptr_ext_<name>``).
    codeguys : list of [filename, content]
        Source files from :func:`generate_c_source`.
    verbose : bool, optional
        Print build diagnostics on failure.

    Returns
    -------
    module : loaded Python extension module exposing ``getptrobj()``.
    tmpdir : str
        Location of the generated sources, useful when a caller wants to
        persist the ``.so`` elsewhere (e.g. a cross-session disk cache).
    """
    import os
    import sys
    import time
    import random
    import importlib.machinery
    from importlib._bootstrap import _load

    unique_suffix = f"{os.getpid()}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    tmpdir = os.path.join("/tmp", f"{modname}_{unique_suffix}")

    try:
        os.makedirs(tmpdir, exist_ok=True)
    except OSError as e:
        if verbose:
            print(f"Warning: Failed to create tmpdir {tmpdir}: {e}")
        raise RuntimeError(f"Cannot create temporary directory {tmpdir}") from e

    for thing in codeguys:
        filename = thing[0]
        strguy = thing[1]
        with open(os.path.join(tmpdir, filename), "w") as f:
            f.write(strguy)

    process = subprocess.Popen(
        [sys.executable] + "setup.py build_ext --inplace".split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmpdir,
        env=_petsc_build_env(),
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        if verbose:
            print(f"Warning: Build process failed with return code {process.returncode}")
            print(f"stdout: {stdout.decode() if stdout else 'None'}")
            print(f"stderr: {stderr.decode() if stderr else 'None'}")

    def load_dynamic(name, path):
        """Load an extension module from ``path``.

        Borrowed from https://stackoverflow.com/a/55172547 — skips the
        ``sys.modules`` reuse check so we always load a fresh extension.
        """
        loader = importlib.machinery.ExtensionFileLoader(name, path)
        spec = importlib.machinery.ModuleSpec(name=name, loader=loader, origin=path)
        return _load(spec)

    module = None
    if os.path.exists(tmpdir):
        for _file in os.listdir(tmpdir):
            if _file.endswith(".so"):
                module = load_dynamic(modname, os.path.join(tmpdir, _file))
                break
    elif verbose:
        print(f"Warning: tmpdir {tmpdir} does not exist - build process may have failed")

    if module is None:
        raise RuntimeError(
            f"The Underworld extension module does not appear to have been built successfully. "
            f"The generated module may be found at:\n    {str(tmpdir)}\n"
            f"To investigate, you may attempt to build it manually by running\n"
            f"    python3 setup.py build_ext --inplace\n"
            f"from the above directory. Note that a new module will always be written by "
            f"Underworld and therefore any modifications to the above files will not persist into "
            f"your Underworld runtime.\n"
            f"Please contact the developers if you are unable to resolve the issue."
        )

    return module, tmpdir


@timing.routine_timer_decorator
def _createext(
    name,
    mesh: underworld3.discretisation.Mesh,
    callbacks: JITCallbackSet,
    primary_field_list,
    constants_subs_map: Optional[dict] = None,
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    debug_name=None,
):
    """Thin wrapper: generate source, compile, stash in ``_ext_dict[name]``.

    Retained for backwards compatibility with :func:`getext`. New code
    should call :func:`generate_c_source` and :func:`compile_and_load`
    directly — splitting the two phases is what makes cache keys on the
    generated C source possible.
    """
    modname, codeguys, diag = generate_c_source(
        name,
        mesh,
        callbacks,
        primary_field_list,
        constants_subs_map=constants_subs_map,
        verbose=verbose,
        debug=debug,
        debug_name=debug_name,
    )
    module, tmpdir = compile_and_load(modname, codeguys, verbose=verbose)
    _ext_dict[name] = module

    if underworld3.mpi.rank == 0 and verbose:
        randstr = diag["randstr"]
        print(f"Location of compiled module: {str(tmpdir)}")
        print(f"{randstr} Equation count - {diag['eqn_count']}", flush=True)
        print(
            f"{randstr}   {diag['count_residual_sig']:5d}    residuals: "
            f"{diag['residual_equations'][0]}:{diag['residual_equations'][1]}",
            flush=True,
        )
        print(
            f"{randstr}   {diag['count_bc_sig']:5d}   boundaries: "
            f"{diag['boundary_equations'][0]}:{diag['boundary_equations'][1]}",
            flush=True,
        )
        print(
            f"{randstr}   {diag['count_jacobian_sig']:5d}    jacobians: "
            f"{diag['jacobian_equations'][0]}:{diag['jacobian_equations'][1]}",
            flush=True,
        )
        print(
            f"{randstr}   {diag['count_bd_residual_sig']:5d} boundary_res: "
            f"{diag['boundary_residual_equations'][0]}:{diag['boundary_residual_equations'][1]}",
            flush=True,
        )
        print(
            f"{randstr}   {diag['count_bd_jacobian_sig']:5d} boundary_jac: "
            f"{diag['boundary_jacobian_equations'][0]}:{diag['boundary_jacobian_equations'][1]}",
            flush=True,
        )

    return
