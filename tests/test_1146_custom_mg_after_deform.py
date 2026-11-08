"""A solver with a custom-P hierarchy and a FAC patch re-solves after ``mesh.deform`` (#1015).

``mesh.deform`` rebuilds the solver's preconditioner from the options database. Two
things then went wrong on the standard (non-rotated) paths:

- ``_pcmg_still_installed`` took the rebuilt PC for the live hierarchy because it had
  the same level count, so the hierarchy was never installed again;
- the FAC level's ``mg_levels_<l>_pc_type composite`` key, which the standard paths
  leave in the database, came back as a composite with no member PCs, and the
  re-solve stopped with PETSc error 73.

The live check now also requires the finest interpolation to be the one the install
set, and an install drops the FAC keys of any level that has no patch this time. The
hierarchy is installed once over repeat solves and again after the deform, and the
re-solve converges, for Poisson (top-level multigrid) and the Stokes velocity block.

What the re-solve does not get back is the FAC patch: the coarse levels given to
``set_custom_fmg`` do not move with the fine mesh, so after a deform no fine node
coincides with its coarse node, every row reads as patch, and the cover gate declines
it. That is filed separately.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _fault_box():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, regular=False, qdegree=2,
                                             refinement=1)
    levels = base._coarse_level_meshes()[:-1] + [base]
    return levels, base.add_fault(("Flt", np.array([[0.3, 0.45], [0.7, 0.55]])))


def _poisson():
    levels, mesh = _fault_box()
    s = uw.systems.Poisson(mesh)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, levels)
    return s, mesh, lambda: s.snes.getKSP().getPC()


def _stokes():
    levels, mesh = _fault_box()
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("V1146", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1146", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, levels, field_id=0)
    return s, mesh, lambda: s.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC()


@pytest.mark.parametrize("build", [_poisson, _stokes], ids=["poisson", "stokes_velocity"])
def test_the_hierarchy_is_installed_once_and_again_after_a_deform(build, monkeypatch):
    installs = []
    install = custom_mg.CustomMGHierarchy.install

    def _counted(self, solver, verbose=False):
        installs.append(id(self))
        return install(self, solver, verbose=verbose)

    monkeypatch.setattr(custom_mg.CustomMGHierarchy, "install", _counted)
    s, mesh, managed_pc = build()

    s.solve()
    first = managed_pc()
    assert first.getMGSmoother(first.getMGLevels() - 1).getPC().getType() == "composite"
    s.solve()
    s.solve()
    # a repeat solve keeps the live hierarchy: the install is the expensive part (#622)
    assert len(installs) == 1, installs

    c = np.asarray(mesh.X.coords).copy()
    c[:, 0] += 0.01 * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])   # zero on the walls
    mesh.deform(c)
    s.solve()                       # PETSc error 73 here before the fix
    assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
    assert len(installs) == 2, installs
    pc = managed_pc()
    nlev = s._custom_mg_live["nlev"]
    assert pc.getType() == "mg" and pc.getMGLevels() == nlev
    assert pc.getMGInterpolation(nlev - 1).handle == s._custom_mg_live["finest"].handle
