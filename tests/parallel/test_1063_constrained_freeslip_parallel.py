"""Parallel regression test for ``SNES_Stokes_Constrained`` (multiplier free-slip).

The in-saddle Lagrange-multiplier free-slip solver was previously guarded
serial-only. The interior-multiplier reduction is in fact rank-local section
surgery, so the GLOBAL system — and hence the velocity solve and the
gauge-invariant boundary traction — are partition-independent. This test
verifies that, for both isotropic and transverse-isotropic rheology, the
parallel solve reproduces its OWN np=1 answer to a **tight tolerance** (the
residual difference is the parallel reduction order, not the solver):

  * the velocity L2 norm (``int v.v``), and
  * the MEAN-STRIPPED boundary topography (``int (h - hbar)^2`` on the
    constrained boundary), via ``solver.topography(boundary, reference="mean")``.

The raw multiplier ``h`` carries the ``[p,lambda]`` gauge constant — the solver
lands on a partition-dependent representative of it — so only the mean-stripped
(physical) topography is compared. All diagnostics use the parallel-safe
``uw.maths.Integral`` / ``BdIntegral`` reductions (no direct mpi4py).

The np=1 side is computed by running this file's own ``__main__`` as a single-rank
child in the same environment (``serial_reference``), NOT read from a constant
recorded on a developer's machine. That distinction matters twice over here: a
self-referential comparison is the property these tests claim to be testing, and
comparing against a stored constant is what made #564 look like seven instances of
one defect when four of them were the host's gmsh building a different mesh.

What this used to measure, and no longer does. With a rank-local boundary normal
(#564, fixed) the ``[iso]`` velocity L2 was 6.194547793939e-01 at np=1 against
5.982807168537e-01 at np=2 — **3.4 %**, #495's number to every digit — with the
topography 2.4 % adrift; ``[ti]`` moved 0.34 % at np=2 and 2.8 % at np=4. The
default constraint direction is ``mesh.boundary_normal(boundary)``, which was
assembled from each rank's own facets. It now agrees to 1.2e-10 (iso) and 4.1e-10
(ti) across np=1,2,3,4.

Run with:
    mpirun -n 2 python -m pytest --with-mpi \\
      tests/parallel/test_1063_constrained_freeslip_parallel.py
    mpirun -n 4 python -m pytest --with-mpi \\
      tests/parallel/test_1063_constrained_freeslip_parallel.py
"""

import numpy as np
import sympy
import pytest

import underworld3 as uw

from serial_reference import compare, emit, mesh_fingerprint, serial_reference

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(600)]

# Velocity reproduces to the parallel reduction order. The topography is read off
# the multiplier, whose [p,lambda] Schur sub-block grinds into its 200-iteration cap
# on this problem (a converged SNES over a capped inner block — see #564's
# investigation), so it reproduces two to three orders less tightly. Both gates sit
# at least three orders below the pre-fix move they exist to catch (3.4 % and 2.4 %).
_RTOL_VELOCITY = 1.0e-8
_RTOL_TOPOGRAPHY = 1.0e-5


def _solve_diagnostics(kind):
    """Build + solve the constrained free-slip annulus; return partition-independent
    diagnostics ``((L2 velocity, mean-stripped boundary topography), fingerprint)``."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=0.12, qdegree=4)
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    X = mesh.CoordinateSystem.X
    unit_r = mesh.CoordinateSystem.unit_e_0

    st = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    if kind == "ti":
        st.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        st.constitutive_model.Parameters.shear_viscosity_0 = 3.0
        st.constitutive_model.Parameters.shear_viscosity_1 = 1.0
        st.constitutive_model.Parameters.director = sympy.Matrix(
            [np.cos(0.6), np.sin(0.6)])
    else:
        st.constitutive_model = uw.constitutive_models.ViscousFlowModel
        st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.tolerance = 1.0e-8
    st.add_essential_bc((0.0, 0.0), "Lower")
    # NO normal= : this is the path that reads mesh.boundary_normal(), which is what
    # #564 was about. Passing an analytic normal here would test something else.
    h = st.add_constraint_bc(0.0, "Upper")
    st.bodyforce = 1.0e2 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * unit_r
    st.solve(zero_init_guess=True)

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    # gauge-fixed topography via the solver's own accessor (exercises the
    # reference="mean" code path); its L2 over the constrained boundary.
    topo_fn = st.topography("Upper", reference="mean")
    topo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=topo_fn ** 2, boundary="Upper").evaluate()))
    return (L2, topo), mesh_fingerprint(mesh)


def _solve_gauge_diagnostics():
    """Enclosed constrained annulus with an active pressure null space (so the
    automatic pressure gauge fires); return the gauge-relevant diagnostics
    ``((L2 velocity, raw mean pressure, mean-stripped topography), fingerprint)``.

    The enclosed iso problem with an ACTIVE pressure null space exercises the
    automatic pressure gauge (``auto_pressure_gauge``, default on): the constant
    pressure and the constant multiplier are both gauge-free, so without a pin the
    solver lands on a partition-dependent level for each. The auto gauge pins the raw
    PRESSURE reproducibly; the raw multiplier keeps its own gauge freedom, which is
    why topography is read gauge-invariantly via ``reference="mean"``. Velocity is
    physics-neutral under the gauge.
    """
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=0.12, qdegree=4)
    v = uw.discretisation.MeshVariable("Ug", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("Pg", mesh, 1, degree=1)
    X = mesh.CoordinateSystem.X
    unit_r = mesh.CoordinateSystem.unit_e_0

    st = uw.systems.Stokes_Constrained(mesh, velocityField=v, pressureField=p)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.tolerance = 1.0e-10
    st.petsc_use_nullspace = True            # enclosed -> pressure null space active
    st.add_essential_bc((0.0, 0.0), "Lower")  # kill the velocity rotation null space
    st.add_constraint_bc(0.0, "Upper", normal=unit_r)
    st.bodyforce = 1.0e2 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * unit_r
    st.solve(zero_init_guess=True)
    assert st._auto_gauge_callback is not None, "auto pressure gauge should have fired"

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    vol = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    # RAW mean pressure — pinned to ~0 by the auto gauge, partition-reproducible.
    meanP = float(uw.maths.Integral(mesh, p.sym[0]).evaluate()) / vol
    # MEAN-STRIPPED topography (reference="mean") — gauge-invariant, the supported
    # partition-reproducible read of the multiplier.
    topo_fn = st.topography("Upper", reference="mean")
    topoL2 = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=topo_fn ** 2, boundary="Upper").evaluate()))
    return (L2, meanP, topoL2), mesh_fingerprint(mesh)


def test_constrained_raw_gauge_partition_independent():
    """With the automatic pressure gauge on (default), the RAW mean pressure is
    partition-independent (pinned to ~0), the velocity stays bit-identical (the
    gauge is physics-neutral), and the mean-stripped topography is reproducible.
    The raw multiplier level is NOT asserted reproducible — it keeps an
    independent gauge freedom the pressure pin does not touch (use
    reference="mean")."""
    (L2, meanP, topo), fingerprint = _solve_gauge_diagnostics()
    reference = serial_reference(__file__, "gauge")
    # meanP is pinned to ~0 on the gauge boundary, so the claim is an ABSOLUTE one
    # and needs no reference at all — a relative comparison of two numbers near
    # machine zero measures nothing. The other two go against the np=1 run.
    assert abs(meanP) < 1e-6, (
        f"raw mean pressure is not pinned at np={uw.mpi.size}: {meanP!r}")
    compare((L2, topo), {"values": [reference["values"][0], reference["values"][2]],
                         "fingerprint": reference["fingerprint"]},
            rtols=(_RTOL_VELOCITY, _RTOL_TOPOGRAPHY),
            labels=("velocity L2", "mean-stripped topography"),
            fingerprint=fingerprint, what="constrained raw-gauge annulus")


@pytest.mark.parametrize("kind", ["iso", "ti"])
def test_constrained_freeslip_partition_independent(kind):
    """The parallel solve must reproduce its own np=1 answer: velocity to the
    parallel reduction order, and the gauge-fixed (mean-stripped) topography to the
    multiplier solve's reproducibility."""
    values, fingerprint = _solve_diagnostics(kind)
    compare(values, serial_reference(__file__, kind),
            rtols=(_RTOL_VELOCITY, _RTOL_TOPOGRAPHY),
            labels=("velocity L2", "mean-stripped topography"),
            fingerprint=fingerprint, what=f"constrained free-slip [{kind}]")


if __name__ == "__main__":
    # Single-rank child of the parallel run (see serial_reference), and a
    # human-readable recompute: `python <thisfile> {iso,ti,gauge}`.
    import sys
    _kind = sys.argv[1] if len(sys.argv) > 1 else "iso"
    if _kind == "gauge":
        _values, _fingerprint = _solve_gauge_diagnostics()
        emit(_values, _fingerprint)
        uw.mpi.pprint("DIAG_GAUGE " + " ".join(f"{v:.12e}" for v in _values))
    else:
        _values, _fingerprint = _solve_diagnostics(_kind)
        emit(_values, _fingerprint)
        uw.mpi.pprint("DIAG " + " ".join(f"{v:.12e}" for v in _values))
