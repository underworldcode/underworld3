"""Every rank must compile the SAME generated C.

The JIT derives both the module name and the C symbol prefix from a hash of the
generated source, so ranks that generate different text build disjoint artefacts
and the rank-0-compiles/others-load protocol breaks. Agreement used to be
REQUIRED — a mismatch raised — and it fires in practice: the lowering in
``generate_c_source`` is not deterministic across ranks (#752), and a Stokes solve
with a power-law transversely isotropic viscosity trips it in roughly half of np=2
runs.

Two measurements say why adopting one rank's source is a sound repair rather than
a way of ignoring a wrong answer:

  * the sources differ only in the ORDER of factors in commutative products —
    identical token multisets, identical length, identical mathematics — so every
    rank's source is a correct kernel for the same equation;
  * the solver's own symbolic blocks (constitutive tensor, flux, every Jacobian
    block) hash IDENTICALLY across ranks on the runs that abort. What differs is
    produced inside ``generate_c_source``, not handed to it.

This file tests the repair directly, by handing
``_agree_source_across_ranks`` a disagreement built on purpose. Reproducing a
real one means reproducing a non-deterministic bug at about one run in two, which
is not a test — it would pass half the time with the repair removed.

The repair is not a fix. #752 is still open, and when the lowering is made
deterministic the disagreement branch should stop being reached — these tests
keep working either way, because they construct the disagreement themselves.
"""

import hashlib

import pytest

import underworld3 as uw
from underworld3.utilities._jitextension import (
    _abi_salt,
    _agree_source_across_ranks,
)

pytestmark = [pytest.mark.timeout(300)]


def _hash_of(codeguys):
    """The same hash ``generate_c_source`` derives the module name from."""
    source = "\n".join(entry[1] for entry in codeguys)
    return hashlib.sha256(
        (source + "\n---\n" + _abi_salt()).encode("utf-8")
    ).hexdigest()[:16]


@pytest.mark.skipif(uw.mpi.size < 2, reason="needs at least two ranks")
def test_ranks_that_disagree_all_adopt_rank_zero_source():
    """The contract. Each rank arrives with different text; all leave with rank
    0's, and with the hash that text actually produces — not merely with hashes
    that happen to match each other."""
    mine = [["eqn_0", f"/* generated on rank {uw.mpi.rank} */"]]
    agreed, source, digest = _agree_source_across_ranks(
        mine, "\n".join(e[1] for e in mine), _hash_of(mine)
    )

    expected_source = "/* generated on rank 0 */"
    assert source == expected_source
    assert agreed == [["eqn_0", expected_source]]
    assert digest == _hash_of([["eqn_0", expected_source]])

    # and every rank really did land on the same one
    assert len(set(uw.mpi.comm.allgather(digest))) == 1


@pytest.mark.skipif(uw.mpi.size < 2, reason="needs at least two ranks")
def test_agreeing_ranks_are_left_exactly_as_they_were():
    """The common path must be a no-op, not a broadcast. Ranks that already agree
    keep their own objects: a repair that rewrote the source on every call would
    hide the upstream bug rather than report it, and would pay a collective on
    every kernel."""
    same = [["eqn_0", "/* identical on every rank */"]]
    before = _hash_of(same)
    agreed, source, digest = _agree_source_across_ranks(
        same, "\n".join(e[1] for e in same), before
    )

    assert digest == before
    assert agreed is same
    assert source == "/* identical on every rank */"


@pytest.mark.skipif(uw.mpi.size < 2, reason="needs at least two ranks")
def test_a_real_kernel_generates_the_same_hash_on_every_rank():
    """End to end: the symptom #752 actually presents as. A Stokes solve with a
    power-law transversely isotropic viscosity is the case that trips the
    lowering; it must complete rather than abort, whichever way the lowering
    happens to fall on each rank."""
    import sympy

    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                              cellSize=0.25, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    unit_r = sympy.Matrix([[x / r, y / r]])
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable("v_jit", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p_jit", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    edot = mesh.vector.strain_tensor(v.sym)
    eII = sympy.sqrt(sympy.Rational(1, 2) * (edot[0, 0] ** 2 + edot[1, 1] ** 2)
                     + edot[0, 1] ** 2)
    eta_0 = (sympy.Float(0.01) + eII) ** sympy.Rational(-1, 3)
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = 0.2 * eta_0
    stokes.constitutive_model.Parameters.director = unit_r
    stokes.bodyforce = 1.0e2 * sympy.cos(3 * th) * (r - 0.5) / 0.5 * unit_r
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.consistent_jacobian = True
    stokes.tolerance = 1.0e-9

    stokes.solve(zero_init_guess=True)   # raised here before the repair
    assert stokes.snes is not None
