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
    h = st.add_constraint_bc("Upper")
    st.bodyforce = 1.0e2 * sympy.sin(3 * sympy.atan2(X[1], X[0])) * unit_r
    st.solve(zero_init_guess=True)

    L2 = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))
    # gauge-fixed topography via the solver's own accessor (exercises the
    # reference="mean" code path); its L2 over the constrained boundary.
    topo_fn = st.topography("Upper", reference="mean")
    topo = float(np.sqrt(uw.maths.BdIntegral(
        mesh=mesh, fn=topo_fn ** 2, boundary="Upper").evaluate()))
    return L2, topo


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
    # Recompute the serial GOLDEN reference: `python <thisfile> {iso,ti}`.
    import sys
    _kind = sys.argv[1] if len(sys.argv) > 1 else "iso"
    _L2, _topo = _solve_diagnostics(_kind)
    if uw.mpi.rank == 0:
        print(f"DIAG {_L2:.12e} {_topo:.12e}")
