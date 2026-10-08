from typing import Optional
import os
import shutil
import subprocess
from xmlrpc.client import boolean
import sympy
import underworld3
import underworld3.timing as timing
from underworld3.utilities import _jit_graph
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


def _petsc_include_dirs():
    """PETSc's own include directories, from petsc4py's configuration.

    The generated callback header includes ``<petscsystypes.h>`` (#813), so the JIT
    build needs PETSc's headers: ``$PETSC_DIR/include`` and, for ``petscconf.h``,
    ``$PETSC_DIR/$PETSC_ARCH/include``. A conda-forge PETSc puts them on the
    compiler's default path, which is why CI built; a custom PETSc build does not, and
    there every JIT compile failed with "'petscsystypes.h' file not found".
    """
    import petsc4py

    info = petsc4py.get_config()
    candidates = [Path(info["PETSC_DIR"]) / "include"]
    # PETSC_ARCH is empty for a prefix install, which has no arch directory
    if info.get("PETSC_ARCH"):
        candidates.append(Path(info["PETSC_DIR"]) / info["PETSC_ARCH"] / "include")
    return [str(c) for c in candidates if c.is_dir()]


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
        matches what ``generate_c_source()`` expects.
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
        """Lengths of each slot, for the offsets in ``generate_c_source()``."""
        return (len(self.residual), len(self.bcs), len(self.jacobian),
                len(self.bd_residual), len(self.bd_jacobian))


# ============================================================================
# JIT Constants Support
# ============================================================================
#
# UWexpressions that are "constant" (no spatial/field dependencies) are routed
# through PETSc's constants[] array instead of being baked as C literals.
# This allows parameter changes without JIT recompilation.
# ============================================================================

class _JITConstant(sympy.Symbol):
    r"""Symbol subclass that renders as ``constants[i]`` in generated C code.

    Used by the JIT compiler to route constant UWexpressions through PETSc's
    ``PetscDSSetConstants()`` mechanism instead of baking values as C literals.

    Two constants may legitimately share a display name — every
    ``ViscousFlowModel`` calls its viscosity :math:`\eta`, so a two-material
    model has two of them — and each needs its own ``constants[]`` slot. Two
    separate SymPy properties have to hold for that to work, and they are not
    the same property:

    **Identity** — the slot index is in ``_hashable_content``, and the symbol
    is built with ``Symbol.__xnew__`` to bypass SymPy's ``(cls, name)``
    instance cache. Without both, ``Symbol.__new__`` hands back the cached
    instance for that name: the second placeholder IS the first object, and
    setting its ``_ccodestr`` overwrites the first one's, so every occurrence
    renders as one slot.

    **Ordering** — the slot index is also in the NAME. ``_hashable_content``
    does nothing for ``Symbol.sort_key()``, which is derived from the name, so
    two same-named placeholders sort equal; term order inside an ``Add`` then
    falls back to hash order, which is randomised per process. The generated C
    then differs between MPI ranks and ``getext``'s cross-rank hash check
    aborts the run — intermittently, since it depends on the hash seed.

    Identity without ordering is a parallel abort; ordering without identity is
    a silently wrong answer. Keep both. ``tests/test_0103_jit_rampable_constants.py``
    pins each one separately.

    A slot holds a C double, so it is built ``real`` (real and finite) for SymPy's
    simplification (#823). Declared at construction, not by a class handler: SymPy
    shares one assumptions knowledge base between Symbols with the same declared
    assumptions, so a handler's answer could be pre-empted by a plain Symbol's cached
    ``None``.
    """

    __slots__ = ("_const_index", "_ccodestr")

    def __new__(cls, index, name=None):
        # The index leads the name so that sort_key() orders placeholders by
        # slot; see the class docstring on why the name alone is not enough
        # and _hashable_content alone is not either.
        suffix = "" if name is None else f"_{name}"
        obj = sympy.Symbol.__xnew__(cls, f"_jit_const_{index}{suffix}", real=True)
        obj._const_index = index
        obj._ccodestr = f"constants[{index}]"
        return obj

    def _hashable_content(self):
        """Two placeholders differ if their constants[] slot differs."""
        return sympy.Symbol._hashable_content(self) + (self._const_index,)

    def __getnewargs_ex__(self):
        return ((self._const_index, self.name), {})

    def _ccode(self, printer):
        return self._ccodestr


def _extract_constants(all_fns, mesh):
    """The ``constants[]`` manifest of a list of callback expressions: the constant
    atoms their lowered kernels read (``_jit_graph``), ordered as ``_manifest_from``
    orders them.

    Returns
    -------
    list of (int, UWexpression)
        Ordered mapping from constants[] index to UWexpression reference.
    dict
        Mapping from UWexpression to _JITConstant symbol.
    """
    lowered = _jit_graph.lower_callbacks([fn for fn in all_fns if fn is not None], mesh)
    return _manifest_from(_jit_graph.constant_leaves(lowered))


def _manifest_from(constant_exprs):
    """The ``constants[]`` manifest of a set of constant atoms: ``(manifest,
    subs_map)``, slots ordered by name, then creation order."""
    if not constant_exprs:
        return [], {}

    # Sort by the user-given symbol name, not ``str(expr)`` — ``__str__`` on a
    # UWexpression returns the current *value*, which shuffles the index
    # assignment whenever a value changes. ``.name`` is stable.
    #
    # Two constants can legitimately SHARE a name: every ViscousFlowModel calls
    # its viscosity \eta, so a model with two of them has two \eta constants.
    # ``instance_number`` (creation order, identical on every rank running the
    # same script) breaks that tie without reintroducing the value into the key.
    # Creation order breaks a name tie. It is identical on every rank of an
    # SPMD run, and unlike the value it does not move when a parameter is
    # ramped — a slot permutation between two solves of the same model would
    # invalidate the JIT cache for no reason.
    sorted_constants = sorted(
        constant_exprs, key=lambda e: (e.name, e.instance_number, _stable_sort_key(e))
    )

    manifest = []
    subs_map = {}
    for i, expr in enumerate(sorted_constants):
        # Use ``expr.name`` (stable) instead of ``str(expr)`` (= current value)
        # so the placeholder symbol's identity is independent of parameter value.
        #
        jit_const = _JITConstant(i, name=expr.name)
        manifest.append((i, expr))
        subs_map[expr] = jit_const

    return manifest, subs_map


def _holds_instance(expr, types):
    """Whether ``expr`` holds a node of ``types``, visiting each node object once."""
    from sympy.tensor.array import NDimArray

    seen = {}
    stack = [expr]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen[id(e)] = e
        if isinstance(e, types):
            return True
        if isinstance(e, (sympy.MatrixBase, NDimArray)):
            stack.extend(e)
        elif isinstance(e, sympy.Basic):
            stack.extend(e.args)
    return False


def _warn_dirac_deltas_dropped(deltas, where, collective=False):
    """Say that ``deltas`` were evaluated as 0 by ``where``, at the user's own call.

    The message names the count, not the delta, so a time loop warns once per call
    site. A ``collective`` caller (the JIT, which every rank runs) warns on rank 0
    only; an evaluation is rank-local and warns wherever it runs.
    """
    if not deltas or (collective and underworld3.mpi.rank != 0):
        return
    import sys
    import warnings

    # the first frame outside the underworld3 package: the user's call
    package = os.path.dirname(underworld3.__file__) + os.sep
    frame, level = sys._getframe(1), 2   # stacklevel 2 is the frame that called this
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame, level = frame.f_back, level + 1
    warnings.warn(
        f"{where}: {len(deltas)} DiracDelta term(s) taken as 0, their value away from "
        f"the zero of the argument. A DiracDelta comes from differentiating a step "
        f"once (a Heaviside or sign of the unknown in a Newton tangent, which then "
        f"leaves out the jump) or a kink twice (Abs(x - a) in a manufactured source). "
        f"A point source has to be applied as a point load.",
        stacklevel=level,
    )


def _without_dirac_deltas(expr, where):
    """``expr`` with every DiracDelta replaced by 0, warning if there were any.

    The rule for every path that turns an expression into numbers: the JIT (through
    its printer), ``uw.function.evaluate`` and the field evaluator (through lambdify).
    A pointwise evaluation cannot carry a distribution.
    """
    if not hasattr(expr, "atoms"):
        return expr
    deltas = sorted(expr.atoms(sympy.DiracDelta), key=sympy.default_sort_key)
    if not deltas:
        return expr
    _warn_dirac_deltas_dropped(deltas, where)
    return expr.xreplace({d: sympy.S.Zero for d in deltas})


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
                # Do NOT pack a zero here. A constants[] slot exists because
                # this expression resolved to a single number when the kernel
                # was COMPILED. If it no longer does, the compiled kernel is
                # structurally wrong for the current model — it reads a scalar
                # where the expression now varies in space — and packing 0.0
                # hands that kernel a zero coefficient. That is silent and
                # catastrophic: a zero diffusivity or viscosity diverges, and
                # nothing says why.
                #
                # The usual cause is a nested atom that has been ramped.
                # `(1 + T**2)**(-m) + 1` is the NUMBER 2 while m is zero, so it
                # banks as one constant; ramp m and it depends on T again, but
                # the kernel still expects a scalar.
                raise RuntimeError(
                    f"constants[] slot {idx} ({uw_expr.name!r}) no longer "
                    f"reduces to a number, so the compiled kernel — which "
                    f"treats it as a scalar constant — is out of date.\n"
                    f"  current content: {str(getattr(uw_expr, '_sym', uw_expr))[:160]}\n"
                    f"This usually means an atom nested inside it has been "
                    f"ramped, and the expression has stopped being constant. "
                    f"Force a rebuild before solving again:\n"
                    f"    solver.is_setup = False\n"
                    f"    solver._needs_function_rewire = True\n"
                    f"To keep a coefficient rampable without this, give it its "
                    f"own atom rather than letting the enclosing expression "
                    f"collapse to a number at compile time — see "
                    f"uw.maths.functions.vanishing."
                ) from None
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
    # Each callback is lowered onto the shared graph of named quantities once
    # (``_jit_graph``, #823); the manifest is the constant leaves of what was lowered.
    lowered_fns = _jit_graph.lower_callbacks(callbacks.flat(), mesh)
    constants_manifest, constants_subs_map = _manifest_from(
        _jit_graph.constant_leaves(lowered_fns))

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
        lowered_fns=lowered_fns,
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

    # All ranks must end up compiling and loading the SAME module: the module
    # name and the C symbol prefix are both derived from `source_hash` below, so
    # ranks that disagree would build disjoint artefacts and the
    # rank-0-compiles/others-load protocol would break.
    #
    # Agreement used to be REQUIRED here, and a mismatch was a hard error. It
    # fires in practice: the lowering above is not yet deterministic across
    # ranks (#752), and a Stokes solve with a power-law transversely isotropic
    # viscosity trips it in roughly half of np=2 runs. What we measured there
    # matters for why this is safe to repair rather than refuse:
    #
    #   * the sources differ only in the ORDER of factors in commutative
    #     products — identical token multisets, identical length, identical
    #     mathematics. Every rank's source is a correct kernel for the same
    #     equation;
    #   * the solver's own symbolic blocks (constitutive tensor, flux, every
    #     Jacobian block) hash IDENTICALLY across ranks on the runs that abort.
    #     What differs is produced inside this function, not handed to it.
    #
    # So the disagreement is about which of several correct spellings to
    # compile, and adopting one of them is enough. Rank 0's is taken, and every
    # rank rehashes from it, which restores the one invariant that matters: one
    # source, one hash, one module.
    #
    # This is a REPAIR, not a fix. The non-determinism upstream is still a bug
    # and still worth finding, which is why it is said out loud rather than
    # papered over silently.
    canonical_codeguys, canonical_source, source_hash = _agree_source_across_ranks(
        canonical_codeguys, canonical_source, source_hash
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
        from mpi4py import MPI as _MPI

        # Disk lookup is cheap and rank-local — every rank checks
        # independently. UW assumes a shared filesystem for the cache dir.
        module = None
        if cache:
            module = _jc.load_module(source_hash, real_modname, constants_manifest)
            if module is not None and verbose and underworld3.mpi.rank == 0:
                print(f"JIT compiled module cached (disk) ... {source_hash}", flush=True)

        # COLLECTIVE decision. The disk lookup above is rank-local — its own
        # comment says so — and the branch below contains a Barrier, so the two
        # must not be joined by a rank-local predicate: one rank finding the
        # published module while another does not leaves the second waiting at
        # a barrier nobody else enters. On the shared filesystems this path
        # exists for, that divergence is ordinary (attribute caching, metadata
        # lag, a write not yet visible), and it is what the barrier is meant to
        # manage rather than something it can assume away.
        #
        # `needs_compile` is a global OR: if ANY rank lacks the module, every
        # rank enters the branch and reaches the barrier. Ranks that already
        # have it keep it and simply take part.
        needs_compile = underworld3.mpi.comm.allreduce(
            module is None, op=_MPI.LOR
        )

        if needs_compile:
            # Cold compile path. With MPI, only rank 0 invokes cc and
            # publishes; the other ranks barrier-wait and then load the
            # freshly-published .so from disk. This avoids the N× cc
            # storm on cold-start under mpirun -np N.
            #
            # Falls back to "every rank compiles" only when the disk
            # cache is disabled (no way to share a build), in which
            # case the wasted cc cost is on the user's chosen path.
            multi_rank = underworld3.mpi.size > 1
            # Also collective, and for the same reason: a rank whose cache
            # directory cannot be resolved would otherwise take the `else`
            # branch — which has no barrier — while its peers wait in one. A
            # global AND makes every rank fall back together, at the cost of
            # each compiling locally, which is the safe direction.
            disk_enabled = underworld3.mpi.comm.allreduce(
                cache and _jc.get_cache_dir() is not None, op=_MPI.LAND
            )

            if multi_rank and disk_enabled:
                if underworld3.mpi.rank == 0 and module is None:
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
                if module is None:
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
                if module is None:
                    module, tmpdir = compile_and_load(
                        real_modname, codeguys_final, verbose=verbose
                    )
                    if verbose and underworld3.mpi.rank == 0:
                        # Tests in test_0004_pointwise_fns parse this exact
                        # prefix to find the per-call build directory.
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


class _Unspellable(Exception):
    """A leaf the kernel has no C for."""


class _Refusal:
    """A leaf that has a meaning but no C in a weak form, with the reason."""

    def __init__(self, message):
        self.message = message


_IP_DERIVATIVE_REFUSAL = _Refusal(
    "derivative of an integration-point "
    "variable has no meaning (the field is defined only at the "
    "quadrature points), so the gradient here would be a silent "
    "zero. This is refused in a WEAK FORM only, where the "
    "discretisation is yours to choose: build the variable with "
    "proxy_location='cells' instead, whose level sets are a "
    "least-squares polynomial per cell and differentiate directly. "
    "uw.function.evaluate() of the same derivative does answer: as "
    "a query it recovers the gradient from a per-cell fit for you."
)


def _leaf_spellings(mesh, primary_field_list):
    """The C a kernel reads for each mesh-variable leaf, keyed by the leaf's class.

    For a 2-D velocity and pressure in the primary arrays: ``V_x -> petsc_u[0]``,
    ``V_y -> petsc_u[1]``, ``P -> petsc_u[2]``, ``V_x_x -> petsc_u_x[0]``, ...,
    ``P_y -> petsc_u_x[5]``. Every field of the mesh is entered first, from the
    auxiliary arrays (``petsc_a``), at its own field's component offset in the DM
    (``_aux_component_offsets``: a dropped variable's field stays in the DM and keeps
    its slots); the primary fields then replace their entries with ``petsc_u``.
    Gradients run to ``cdim``, the embedded dimension, so a manifold mesh's third
    partial is wired too. The gradient of an integration-point variable is a
    ``_Refusal``.
    """
    from underworld3 import VarType

    spellings = {}

    def enter(varlist, prefix, component_offsets=None):
        u_i = 0          # component
        u_x_i = 0        # gradient component
        for var in varlist:
            if component_offsets is not None:
                u_i = component_offsets[var.field_id]
                u_x_i = u_i * mesh.cdim
            if var.vtype == VarType.SCALAR:
                components = [var.fn]
            elif var.vtype in (VarType.VECTOR, VarType.TENSOR, VarType.SYM_TENSOR,
                               VarType.MATRIX):
                components = list(var.sym_1d)
            else:
                raise RuntimeError(
                    f"Unsupported type {var.vtype} for code generation. "
                    f"Please contact developers.")
            ip = getattr(var, "is_integration_point", False)
            for component in components:
                spellings[type(component)] = f"{prefix}[{u_i}]"
                u_i += 1
                for ind in range(mesh.cdim):
                    # _diff[ind] is the gradient component's class
                    spellings[component._diff[ind]] = (
                        _IP_DERIVATIVE_REFUSAL if ip else f"{prefix}_x[{u_x_i}]")
                    u_x_i += 1

    enter(_stable_sorted(mesh.vars.values()), "petsc_a",
          component_offsets=_aux_component_offsets(mesh))
    enter(primary_field_list, "petsc_u")
    return spellings


def _spell_leaf(leaf, spellings, constants_subs_map):
    """The C for one leaf: a constants[] slot, a field value or gradient
    (``spellings``), a coordinate or boundary normal, or a symbol that names its own
    C (the time, ``petsc_t``). Raises ``_Unspellable`` for anything else."""
    from sympy.core.function import AppliedUndef
    from sympy.vector.scalar import BaseScalar

    placeholder = constants_subs_map.get(leaf) if constants_subs_map else None
    if placeholder is not None:
        return placeholder._ccodestr
    if isinstance(leaf, BaseScalar):
        # the mesh names its coordinates; a fresh instance, or a UWCoordinate that
        # SymPy's cache returned for its equal base scalar, is named from its index
        # and system
        try:
            text = leaf._ccodestr
        except AttributeError:
            text = None
        if isinstance(text, str):
            return text
        idx, system = leaf._id[0], str(leaf._id[1])
        return f"petsc_n[{idx}]" if "Gamma" in system else f"petsc_x[{idx}]"
    if isinstance(leaf, AppliedUndef):
        entry = spellings.get(type(leaf))
        if entry is None:
            raise _Unspellable(leaf)
        if isinstance(entry, _Refusal):
            raise RuntimeError(f"{type(leaf).__name__}: {entry.message}")
        return entry
    text = getattr(leaf, "_ccodestr", None)
    if isinstance(text, str) and hasattr(leaf, "_ccode"):
        return text
    raise _Unspellable(leaf)


def _unconvertible_message(symbols, index, fn_original):
    details = []
    for sym in symbols:
        detail = f"  - {sym} (type: {type(sym).__name__})"
        if hasattr(sym, "units"):
            detail += f" [has units: {sym.units}]"
        if hasattr(sym, "value"):
            detail += f" [value: {sym.value}]"
        details.append(detail)
    return (
        f"\n{'=' * 70}\n"
        f"JIT COMPILATION ERROR: Expression contains unconvertible symbols\n"
        f"{'=' * 70}\n\n"
        f"The following symbols could not be converted to C code:\n"
        + "\n".join(details) + "\n\n"
        f"This usually means:\n"
        f"  1. A UWexpression or UWQuantity was not properly expanded\n"
        f"  2. An arithmetic operation failed (e.g., Matrix * UWexpression)\n"
        f"  3. A symbolic function is missing from the expression tree\n"
        f"  4. A field that belongs to another mesh, or to no mesh\n\n"
        f"Expression index: {index}\n"
        f"Original expression: {fn_original}\n\n"
        f"TIP: Check that all expression operations (*, /, +, -) produce\n"
        f"valid SymPy expressions. For example, ensure scalar * Matrix\n"
        f"and not Matrix * scalar when using UWexpression objects.\n"
        f"{'=' * 70}"
    )


def _print_kernel(printer, temporaries, outputs, out):
    """The C body of one kernel: a ``const double`` per temporary, in order, then
    the outputs. A SymPy function the printer cannot write returns its
    ``// Not supported in C:`` text, which the caller refuses."""
    lines = []
    for t, body in temporaries:
        code = printer.doprint(body)
        if code.startswith("// Not supported in C:"):
            return code
        lines.append(f"const double {t._ccodestr} = {code};")
    lines.append(printer.doprint(outputs, out))
    return "\n".join(lines)


@timing.routine_timer_decorator
def _aux_component_offsets(mesh):
    """Component offset of every field of the mesh DM, keyed by field id.

    Read from the DM itself, not from ``mesh.vars``: a MeshVariable that
    was dropped and collected leaves its PETSc field in the DM (a DMPlex
    cannot shed a field), and PETSc lays the auxiliary arrays out over
    ALL fields in field order. The offsets therefore have to count the
    orphaned fields too.
    """
    offsets = {}
    total = 0
    for field_id in range(mesh.dm.getNumFields()):
        fe, _label = mesh.dm.getField(field_id)
        offsets[field_id] = total
        total += fe.getNumComponents()
    return offsets


def _agree_source_across_ranks(canonical_codeguys, canonical_source, source_hash):
    """Make every rank compile the SAME generated C, and say so if they did not.

    Returns the (possibly replaced) ``(codeguys, source, hash)``. Serial runs and
    runs where the ranks already agree are returned untouched, so the common path
    costs one ``allgather`` of a 16-character string.

    See the call site for why adopting one rank's source is a sound repair rather
    than papering over a wrong answer. Separated out so the repair can be tested
    directly — forcing a real disagreement through the JIT means reproducing a
    non-deterministic bug, which is not a test.
    """
    import hashlib          # module-local in generate_c_source too

    if underworld3.mpi.size <= 1:
        return canonical_codeguys, canonical_source, source_hash

    all_hashes = underworld3.mpi.comm.allgather(source_hash)
    if all(h == source_hash for h in all_hashes):
        return canonical_codeguys, canonical_source, source_hash

    canonical_codeguys = underworld3.mpi.comm.bcast(canonical_codeguys, root=0)
    canonical_source = "\n".join(entry[1] for entry in canonical_codeguys)
    source_hash = hashlib.sha256(
        (canonical_source + "\n---\n" + _abi_salt()).encode("utf-8")
    ).hexdigest()[:16]
    underworld3.mpi.pprint(
        f"[jit] WARNING: generated C differed across ranks "
        f"({sorted(set(all_hashes))}); adopted rank 0's source so every rank "
        f"compiles the same module. The kernels are mathematically identical — "
        f"see issue #752 for the upstream non-determinism."
    )
    return canonical_codeguys, canonical_source, source_hash


def generate_c_source(
    name,
    mesh: underworld3.discretisation.Mesh,
    callbacks: JITCallbackSet,
    primary_field_list,
    constants_subs_map: Optional[dict] = None,
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    debug_name=None,
    lowered_fns=None,
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
        Mapping from UWexpression → ``_JITConstant`` placeholder; built from the
        lowered callbacks when not given.
    lowered_fns : list of sympy.Matrix, optional
        The callbacks lowered onto the shared graph (``_jit_graph.lower_callbacks``),
        one per entry of ``callbacks.flat()``; lowered here when not given. Each
        kernel is emitted as one C temporary per distinct computation, then its
        outputs.

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

    if lowered_fns is None:
        lowered_fns = _jit_graph.lower_callbacks(fns, mesh)
    if constants_subs_map is None:
        _, constants_subs_map = _manifest_from(_jit_graph.constant_leaves(lowered_fns))

    # The C a kernel reads for each leaf of the graph: an explicit map, built for
    # this compile, instead of C names patched onto the field classes (#823).
    spellings = _leaf_spellings(mesh, primary_field_list)

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

    # A DiracDelta is printed as 0, its value away from the zero of its argument, by
    # the rule every evaluation path shares (_without_dirac_deltas). Done in the
    # printer so that one made while lowering (cse, temporaries) is caught too.
    dropped_deltas = []

    def _print_DiracDelta(expr, **kwargs):
        dropped_deltas.append(expr)
        return "0.0"

    printer._print_DiracDelta = _print_DiracDelta

    # SymPy's code printer rewrites re(UnevaluatedExpr(<real>)) with a whole-tree
    # replace on every kernel it prints (2.9 s of the notch C generation). Field values
    # and coordinates are real (#823), so a kernel seldom holds any re() at all; the
    # rewrite runs only when one is there.
    handle_unevaluated = printer._handle_UnevaluatedExpr

    def _handle_UnevaluatedExpr(expr):
        return handle_unevaluated(expr) if _holds_instance(expr, sympy.re) else expr

    printer._handle_UnevaluatedExpr = _handle_UnevaluatedExpr

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
    for index, fn_original in enumerate(fns):
        unspellable = []

        def spell(leaf):
            try:
                return _spell_leaf(leaf, spellings, constants_subs_map)
            except _Unspellable:
                unspellable.append(leaf)
                return f"?{sympy.srepr(leaf)}"

        temporaries, fn = _jit_graph.emit(lowered_fns[index], spell)
        if unspellable:
            raise RuntimeError(_unconvertible_message(
                _stable_sorted(set(unspellable)), index, fn_original))

        if verbose:
            # the kernel as mathematics (named quantities as nodes); the C is in the header
            print("Processing JIT {:4d} / {}".format(index, lowered_fns[index]))

        out = sympy.MatrixSymbol("out", *fn.shape)
        eqn = ("eqn_" + str(index), _print_kernel(printer, temporaries, fn, out))

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

    _warn_dirac_deltas_dropped(dropped_deltas, "JIT", collective=True)

    MODNAME = "fn_ptr_ext_" + str(name)

    # JIT compile flags for the generated kernels. -std=c99 is always first (the
    # generated code relies on it); UW3_JIT_CFLAGS replaces the rest.
    #   -O3   kernel run-time speed.
    #   -g0   drops the debug info the base Python CFLAGS inject via sysconfig:
    #         overhead, and a memory hog on huge expressions.
    #   -fno-math-errno  the kernels never read errno. gcc and clang on Linux
    #         default to -fmath-errno, which makes each sqrt, pow and exp call a
    #         side effect that cannot be merged with a repeat of the same call; the
    #         flag cut a viscoplastic Jacobian's callbacks to a third on gcc 14, with
    #         bit-identical results for the C SymPy prints (#834). clang targeting
    #         macOS already defaults to it.
    # For a very large expression whose -O3 compile is slow or OOM-killed, use a
    # lower level, e.g. UW3_JIT_CFLAGS="-O1 -g0 -fno-math-errno". The override
    # replaces all three, so a compiler that rejects one (nvc rejects -g0 and
    # -fno-math-errno) can still be used.
    default_flags = ["-O3", "-g0", "-fno-math-errno"]
    _jit_cflags_env = os.environ.get("UW3_JIT_CFLAGS")
    chosen_flags = _jit_cflags_env.split() if _jit_cflags_env is not None else default_flags
    extra_compile_args = ["-std=c99", *chosen_flags]
    if verbose:
        print(
            f"JIT compile flags: {extra_compile_args}"
            f"{' (from UW3_JIT_CFLAGS)' if _jit_cflags_env is not None else ' (default)'}",
            flush=True,
        )

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
    extra_compile_args={EXTRA_COMPILE_ARGS},
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
""".format(
        NAME=MODNAME,
        HEADERS=list(_stable_sorted(underworld3._incdirs.keys())) + _petsc_include_dirs(),
        LIBDIRS=list(_stable_sorted(underworld3._libdirs.keys())),
        LIBFILES=list(_stable_sorted(underworld3._libfiles.keys())),
        EXTRA_COMPILE_ARGS=extra_compile_args,
    )
    codeguys.append(["setup.py", setup_py_str])

    residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    bd_residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], const PetscReal petsc_n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    bd_jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[],  const PetscReal petsc_n[],PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"

    # Create header top content.
    h_str = """
#include <petscsystypes.h>
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
        debug_str = debugging_text(randstr, fns[fn_counter], "  res", fn_counter) if debug else ""
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], residual_sig, eqn[1], debug_str
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bc_sig

    # The bcs have the same signature as the residuals (at present)
    # but we leave this separate in case it changes in later PETSc implementations

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  ebc", fn_counter) if debug else ""
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], residual_sig, eqn[1], debug_str
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_jacobian_sig

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  jac", fn_counter) if debug else ""

        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], jacobian_sig, eqn[1], debug_str
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], jacobian_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bd_residual_sig
    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text_bd(randstr, fns[fn_counter], "bdres", fn_counter) if debug else ""
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], bd_residual_sig, eqn[1], debug_str
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], bd_residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bd_jacobian_sig
    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text_bd(randstr, fns[fn_counter], "bdjac", fn_counter) if debug else ""
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], bd_jacobian_sig, eqn[1], debug_str
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
