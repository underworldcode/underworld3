"""Parallel regression: solver.boundary_flux is partition-independent, including when
the flux boundary is CUT by the partition (np=4 on this box splits the bottom).

The nodal reaction is PETSc's volume FEM residual (rock-solid); a boundary node shared
across a partition cut holds only each rank's partial contribution (DM overlap=0), and the
complete reaction is assembled by summing the partials across ranks by coordinate. This
test checks that the surface heat flux — and its BdIntegral — reproduce the serial
reference at np=2 and np=4.

Run:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1065_boundary_flux_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1065_boundary_flux_parallel.py
"""
import numpy as np
import sympy
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(180)]

# SERIAL reference: BdIntegral of the flux field over Bottom. `python <thisfile>`.
GOLDEN_BDFLUX = -1.731543e-01


def _flux_diagnostics(res=48):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Tp", mesh, 1, degree=2)
    q = uw.discretisation.MeshVariable("qp", mesh, 1, degree=1, continuous=True)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Left")
    poisson.add_dirichlet_bc(0.0, "Right")
    poisson.add_dirichlet_bc(sympy.Matrix([sympy.sin(sympy.pi * x)]), "Top")
    poisson.tolerance = 1e-11
    poisson.petsc_options["snes_type"] = "ksponly"
    poisson.solve()

    xs, flux = poisson.boundary_flux("Bottom")
    poisson.boundary_flux_field("Bottom", q)
    bd_q = float(uw.maths.BdIntegral(mesh=mesh, fn=q.sym[0], boundary="Bottom").evaluate())

    # gather + dedup for a whole-boundary relL2 vs analytic (on rank 0, then bcast)
    comm = uw.mpi.comm
    gx = comm.gather(np.asarray(xs).reshape(-1, 2), root=0)
    gf = comm.gather(np.asarray(flux).reshape(-1), root=0)
    relL2 = None
    if uw.mpi.rank == 0:
        seen = {}
        for xb, fb in zip(gx, gf):
            for xc, fc in zip(xb, fb):
                seen[(round(float(xc[0]), 9),)] = (xc[0], fc)
        X = np.array([v[0] for v in seen.values()])
        F = np.array([v[1] for v in seen.values()])
        q_an = np.pi * np.sin(np.pi * X) / np.sinh(np.pi)
        c = np.dot(F, q_an) / (np.linalg.norm(F) * np.linalg.norm(q_an))
        F = F if c >= 0 else -F
        relL2 = float(np.linalg.norm(F - q_an) / np.linalg.norm(q_an))
    return bd_q, comm.bcast(relL2, root=0)


def test_boundary_flux_partition_independent():
    """boundary_flux reproduces the serial reference at np=2 and np=4 (flux boundary cut
    at np=4): both the collective BdIntegral of the flux field and the whole-boundary
    accuracy vs analytic."""
    bd_q, relL2 = _flux_diagnostics(res=48)
    assert np.isclose(bd_q, GOLDEN_BDFLUX, rtol=1e-5, atol=0), (
        f"BdIntegral flux differs serial vs np={uw.mpi.size}: {GOLDEN_BDFLUX} vs {bd_q}")
    assert relL2 < 0.01, f"heat flux relL2 vs analytic {relL2:.4f} too large at np={uw.mpi.size}"


if __name__ == "__main__":
    _b, _r = _flux_diagnostics()
    if uw.mpi.rank == 0:
        print(f"DIAG_FLUX bd_q={_b:.9e} relL2={_r:.4f}")
