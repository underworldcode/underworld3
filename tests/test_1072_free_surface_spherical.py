"""3D FreeSurface on a spherical shell: the end-to-end loop must run and produce
physically sensible topography (guards the dimension-general surface machinery:
owned-facet trace-mass gauge, P1-projected sigma_nn recovery, radial deform).

The quantitative benchmarking (analytic Y_lm shell rate, convergence of the
h_inf modal bias, 3D parallel) is the review-team's scope — this test pins the
CAPABILITY: construction, one solve/advance cycle, finite mean-free h_inf, and
the explicit refusal of the 2D-only features.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _shell_stokes(cell=0.35):
    mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.547,
                                     cellSize=cell, qdegree=3)
    x, y, z = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    rhat = sympy.Matrix([[x / r, y / r, z / r]])
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    blob = sympy.exp(-(((x - 0.75) ** 2 + y ** 2 + z ** 2) / 0.05))
    stokes.bodyforce = 50.0 * blob * rhat.T
    stokes.add_essential_bc((0.0, 0.0, 0.0), "Lower")
    stokes.tolerance = 1.0e-5
    return mesh, stokes, rhat


def test_freesurface_spherical_shell_end_to_end():
    """Construction + one full solve/advance on the shell; h_inf finite and
    mean-free; the surface responds toward equilibrium (|h| grows from flat
    under the one-sided load and stays bounded by |h_inf|)."""
    mesh, stokes, rhat = _shell_stokes()
    fs = uw.systems.FreeSurface(stokes, "Upper", buoyancy_scale=50.0, normal=rhat)
    fs.solve()
    h_inf = np.asarray(fs._h_inf)
    assert np.isfinite(h_inf).all(), "3D h_inf recovery produced non-finite values"
    assert abs(fs._surface_mean(h_inf)) < 1.0e-8 * (np.abs(h_inf).max() + 1e-30), \
        "h_inf datum is not mean-free under the trace-mass gauge"
    assert np.abs(h_inf).max() > 1.0e-4, "no topographic response to the load"
    fs.advance(fs.estimate_dt(advect_scale=10.0))
    shape = fs._current_shape()
    assert np.isfinite(shape).all()
    assert 0.0 < np.abs(shape).max() <= 1.5 * np.abs(h_inf).max(), \
        "surface did not move toward (or overshot) equilibrium"


def test_freesurface_spherical_refuses_2d_only_features():
    """The 2D-only features fail loudly at construction in 3D, not silently."""
    mesh, stokes, rhat = _shell_stokes()
    with pytest.raises(NotImplementedError, match="tangential"):
        uw.systems.FreeSurface(stokes, "Upper", normal=rhat, tangent_advect="shape")
    with pytest.raises(NotImplementedError, match="filter"):
        uw.systems.FreeSurface(stokes, "Upper", normal=rhat, surface_filter=10)
