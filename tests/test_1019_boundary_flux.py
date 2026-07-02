"""Consistent Boundary Flux (solver.boundary_flux) — scalar surface heat flux.

The essential-BC reaction of a diffusion solve, de-smeared by the boundary mass, is the
consistent surface flux -k dT/dn (Gresho et al.); its boundary mean is the Nusselt
number. Validated against a harmonic manufactured solution with a known analytic flux.
The nodal reaction is PETSc's (rock-solid) volume FEM residual; its complete value at a
partition-cut boundary node is assembled by summing each rank's partial contribution by
coordinate, so the flux is partition-independent (see the parallel test).
"""
import numpy as np
import sympy
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _heatflux_diagnostics(res=48):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Tbf", mesh, 1, degree=2)
    q = uw.discretisation.MeshVariable("qbf", mesh, 1, degree=1, continuous=True)
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

    # general API: boundary flux + field hand-off
    xs, flux = poisson.boundary_flux("Bottom")               # lumped, mean kept
    poisson.boundary_flux_field("Bottom", q)
    # field is symbolically usable
    bd_q = float(uw.maths.BdIntegral(mesh=mesh, fn=q.sym[0], boundary="Bottom").evaluate())

    xc = np.asarray(xs)[:, 0] if len(xs) else np.zeros(0)
    q_an = np.pi * np.sin(np.pi * xc) / np.sinh(np.pi)       # analytic outward flux
    return np.asarray(flux), q_an, bd_q


def test_boundary_flux_scalar_heatflux_serial():
    """Surface heat flux reproduces the analytic flux to high accuracy, and its mean is
    the (analytic) Nusselt number — NOT removed."""
    flux, q_an, bd_q = _heatflux_diagnostics(res=48)
    corr = np.dot(flux, q_an) / (np.linalg.norm(flux) * np.linalg.norm(q_an))
    fa = flux if corr >= 0 else -flux
    relL2 = np.linalg.norm(fa - q_an) / np.linalg.norm(q_an)
    assert abs(corr) > 0.999, f"heat flux corr {corr:.4f} too low"
    assert relL2 < 0.01, f"heat flux relL2 vs analytic {relL2:.4f} too large"
    # physical MEAN flux preserved (Nusselt): |mean| ≈ 2/sinh(pi)
    assert np.isclose(abs(fa.mean()), 2.0 / np.sinh(np.pi), rtol=0.02), (
        f"mean flux {fa.mean():.4f} != Nusselt {2.0/np.sinh(np.pi):.4f}")
    assert abs(bd_q) > 0.0                                    # field populated + usable


if __name__ == "__main__":
    _f, _a, _b = _heatflux_diagnostics()
    c = np.dot(_f, _a) / (np.linalg.norm(_f) * np.linalg.norm(_a))
    print(f"corr={abs(c):.4f} relL2={np.linalg.norm((_f if c>=0 else -_f)-_a)/np.linalg.norm(_a):.4f}")
