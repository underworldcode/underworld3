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
from mpi4py import MPI

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(180)]

# SERIAL reference: BdIntegral of the flux field over Bottom. `python <thisfile>`.
GOLDEN_BDFLUX = -1.731543e-01
ANALYTIC_DIRECT_INTEGRAL = -2.0 / np.sinh(np.pi)
GOLDEN_DIRECT_INTEGRAL = -1.731790673330021e-01


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
    direct_integral = poisson.boundary_flux_integral("Bottom")

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
    return bd_q, direct_integral, comm.bcast(relL2, root=0)


def test_boundary_flux_partition_independent():
    """boundary_flux reproduces the serial reference at np=2 and np=4 (flux boundary cut
    at np=4): both the collective BdIntegral of the flux field and the whole-boundary
    accuracy vs analytic."""
    bd_q, direct_integral, relL2 = _flux_diagnostics(res=48)
    assert np.isclose(bd_q, GOLDEN_BDFLUX, rtol=1e-5, atol=0), (
        f"BdIntegral flux differs serial vs np={uw.mpi.size}: {GOLDEN_BDFLUX} vs {bd_q}")
    assert relL2 < 0.01, f"heat flux relL2 vs analytic {relL2:.4f} too large at np={uw.mpi.size}"
    assert np.isclose(
        direct_integral, GOLDEN_DIRECT_INTEGRAL, rtol=1.0e-10, atol=0.0
    ), (
        "Direct reaction integral differs from the serial reference at "
        f"np={uw.mpi.size}: {GOLDEN_DIRECT_INTEGRAL} vs {direct_integral}"
    )
    assert np.isclose(
        direct_integral, ANALYTIC_DIRECT_INTEGRAL, rtol=1.0e-7, atol=0.0
    )


def _uniform_flux_3d_error(degree, mass):
    """Maximum pointwise error for unit-cube conduction on the Bottom trace."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.45,
        regular=True,
        qdegree=3,
    )
    temperature = uw.discretisation.MeshVariable(
        f"Tbf3dp_p{degree}", mesh, 1, degree=degree
    )
    poisson = uw.systems.Poisson(mesh, u_Field=temperature)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.tolerance = 1.0e-11
    poisson.petsc_options["snes_type"] = "ksponly"
    poisson.solve()
    _xs, flux = poisson.boundary_flux("Bottom", mass=mass)
    local_error = float(np.max(np.abs(np.asarray(flux) + 1.0))) if len(flux) else 0.0
    return uw.mpi.comm.allreduce(local_error, op=MPI.MAX)


@pytest.mark.parametrize(("degree", "mass"), ((1, "lumped"), (2, "auto")))
def test_boundary_flux_3d_pointwise_uniform_partition_independent(degree, mass):
    """P1 and P2 constant-flux recovery is pointwise exact on every MPI partition."""
    max_error = _uniform_flux_3d_error(degree, mass)
    assert max_error < 1.0e-8, (
        f"P{degree} pointwise flux error {max_error:.3e} at np={uw.mpi.size}"
    )


def test_boundary_flux_degree3_partition_independent():
    """#459 at np >= 2: a degree-3 trace keeps one coordinate per edge-interior node,
    and the per-node coordinate build is COLLECTIVE — ranks owning none of the flux
    boundary must still participate. T = 1 - y is exact, so every trace node on every
    partition reads the exact unit flux."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=4)
    T = uw.discretisation.MeshVariable("T459p", mesh, 1, degree=3)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()
    for wall, sign in (("Top", -1.0), ("Bottom", +1.0)):
        _xs, flux = poisson.boundary_flux(wall)
        local_error = float(np.max(np.abs(np.asarray(flux) - sign))) if len(flux) else 0.0
        max_error = uw.mpi.comm.allreduce(local_error, op=MPI.MAX)
        assert max_error < 1e-3, (
            f"{wall}: degree-3 flux error {max_error:.3e} at np={uw.mpi.size}")


if __name__ == "__main__":
    _b, _i, _r = _flux_diagnostics()
    if uw.mpi.rank == 0:
        print(f"DIAG_FLUX bd_q={_b:.9e} relL2={_r:.4f}")
