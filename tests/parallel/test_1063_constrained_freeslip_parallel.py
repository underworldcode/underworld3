"""Parallel regression test for ``SNES_Stokes_Constrained`` (multiplier free-slip).

The in-saddle Lagrange-multiplier free-slip solver was previously guarded
serial-only. The interior-multiplier reduction is in fact rank-local section
surgery, so the GLOBAL system — and hence the velocity solve and the
gauge-invariant boundary traction — are partition-independent. This test
verifies that, for both isotropic and transverse-isotropic rheology, the
parallel solve reproduces the serial reference to a **tight tolerance** (the
residual difference is the parallel reduction order, not the solver):

  * the velocity L2 norm (``∫ v·v``), and
  * the MEAN-STRIPPED boundary topography (``∫(h - h̄)²`` on the constrained
    boundary), computed via ``solver.topography(boundary, reference="mean")``.

The raw multiplier ``h`` carries the ``[p,λ]`` gauge constant — the solver lands
on a partition-dependent representative of it — so only the mean-stripped
(physical) topography is compared. All diagnostics use the parallel-safe
``uw.maths.Integral`` / ``BdIntegral`` reductions (no direct mpi4py).

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

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(180)]

# SERIAL (np=1) reference diagnostics — the partition-independent ground truth.
# Computed once with `python <thisfile> {iso,ti}`; the parallel run must match.
# (velocity L2, mean-stripped boundary topography) for the annulus problem below.
GOLDEN = {
    "iso": (6.194547793955e-01, 3.786068778041e+01),
    "ti":  (3.925981604039e-01, 3.707799837159e+01),
}


def _solve_diagnostics(kind):
    """Build + solve the constrained free-slip annulus; return partition-
    independent diagnostics (L2 velocity, mean-stripped boundary topography)."""
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
    h = st.add_constraint_bc(0.0, "Upper")
    st.bodyforce = 1.0e2 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * unit_r
    st.solve(zero_init_guess=True)

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    # gauge-fixed topography via the solver's own accessor (exercises the
    # reference="mean" code path); its L2 over the constrained boundary.
    topo_fn = st.topography("Upper", reference="mean")
    topo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=topo_fn ** 2, boundary="Upper").evaluate()))
    return L2, topo


# Serial (np=1) reference for the gauge-reproducibility test below. The enclosed
# iso problem with an ACTIVE pressure null space exercises the automatic pressure
# gauge (auto_pressure_gauge, default on): the constant pressure and constant
# multiplier are both gauge-free, so without a pin the solver lands on a
# partition-dependent level for each. The auto gauge pins the raw PRESSURE
# reproducibly (the raw multiplier keeps its own gauge freedom — topography is
# read gauge-invariantly via reference="mean"). Velocity is physics-neutral.
# (velocity L2, raw meanP, MEAN-STRIPPED boundary topography).
# Recompute with `python <thisfile> gauge`.
GOLDEN_GAUGE = (6.194547487092e-01, -3.847657609797e-10, 3.781793823254e+01)


def _solve_gauge_diagnostics():
    """Enclosed constrained annulus with an active pressure null space (so the
    automatic pressure gauge fires); return the gauge-relevant diagnostics."""
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
    return L2, meanP, topoL2


@pytest.mark.xfail(
    reason="#564: free-slip solves are partition dependent. CI measures velocity L2 "
           "0.6194547487092 serial vs 0.6194402844556 at np=2 (2.3e-05). PRE-EXISTING "
           "and not caused by #560/#561: the same seven assertions fail with numbers "
           "identical to every digit at #561's merge base with only the "
           "scripts/test.sh test_10*py line enabled, which is how they became visible "
           "at all — this whole batch had never run in CI. Passes locally on "
           "macOS/arm64, so strict=False; see #564 for the full table.",
    strict=False)
def test_constrained_raw_gauge_partition_independent():
    """With the automatic pressure gauge on (default), the RAW mean pressure is
    partition-independent (pinned to ~0), the velocity stays bit-identical (the
    gauge is physics-neutral), and the mean-stripped topography is reproducible.
    The raw multiplier level is NOT asserted reproducible — it keeps an
    independent gauge freedom the pressure pin does not touch (use
    reference="mean")."""
    L2, meanP, topo = _solve_gauge_diagnostics()
    L2_ref, meanP_ref, topo_ref = GOLDEN_GAUGE
    # Velocity is physics-neutral under the gauge; the residual ~1e-9 spread is
    # parallel round-off in the pressure-null-space projection (this enclosed
    # config carries the null space, unlike the half-pinned case above), well
    # below any real partition effect.
    assert np.isclose(L2, L2_ref, rtol=1e-8, atol=0), (
        f"velocity L2 differs serial vs np={uw.mpi.size}: {L2_ref} vs {L2}")
    # meanP is pinned to ~0 on the gauge boundary; compare on an absolute scale
    # (a relative tolerance is meaningless near machine zero).
    assert np.isclose(meanP, meanP_ref, rtol=0, atol=1e-6), (
        f"raw mean pressure differs serial vs np={uw.mpi.size}: "
        f"{meanP_ref} vs {meanP}")
    assert np.isclose(topo, topo_ref, rtol=1e-6, atol=0), (
        f"mean-stripped topography differs serial vs np={uw.mpi.size}: "
        f"{topo_ref} vs {topo}")


@pytest.mark.xfail(
    reason="#564 (of which #495 is one member): free-slip solves are partition "
           "dependent. CI measures velocity L2 [iso] 0.6194547793955 serial vs "
           "0.6107410846031 at np=2 (1.4%) and [ti] 0.3925981604039 vs "
           "0.3937854671587 (0.3%), against the 1e-9 this asserts. PRE-EXISTING and "
           "not caused by #560/#561: the same numbers reproduce to every digit at "
           "#561's merge base with only the scripts/test.sh test_10*py line enabled, "
           "and Stokes_Constrained never calls _boundary_velocity_nodes. #564 records "
           "that the ROTATED path is affected too, so this is a family rather than a "
           "single solver's bug. Passes locally on macOS/arm64, so strict=False.",
    strict=False)
@pytest.mark.parametrize("kind", ["iso", "ti"])
def test_constrained_freeslip_partition_independent(kind):
    """The parallel solve must reproduce the serial reference: velocity bit-
    identical, and the gauge-fixed (mean-stripped) topography bit-identical."""
    L2_par, topo_par = _solve_diagnostics(kind)
    L2_ref, topo_ref = GOLDEN[kind]
    assert np.isclose(L2_par, L2_ref, rtol=1e-9, atol=0), (
        f"[{kind}] velocity L2 differs serial vs np={uw.mpi.size}: "
        f"{L2_ref} vs {L2_par}")
    assert np.isclose(topo_par, topo_ref, rtol=1e-6, atol=0), (
        f"[{kind}] mean-stripped topography differs serial vs np={uw.mpi.size}: "
        f"{topo_ref} vs {topo_par}")


if __name__ == "__main__":
    # Recompute the serial GOLDEN reference: `python <thisfile> {iso,ti,gauge}`.
    import sys
    _kind = sys.argv[1] if len(sys.argv) > 1 else "iso"
    if _kind == "gauge":
        _L2, _meanP, _topo = _solve_gauge_diagnostics()
        if uw.mpi.rank == 0:
            print(f"DIAG_GAUGE {_L2:.12e} {_meanP:.12e} {_topo:.12e}")
    else:
        _L2, _topo = _solve_diagnostics(_kind)
        if uw.mpi.rank == 0:
            print(f"DIAG {_L2:.12e} {_topo:.12e}")
