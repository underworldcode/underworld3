"""A custom-P hierarchy's state does not outlive a change to the solver or the mesh.

- **#1019.** The FAC install wrote its smoother keys (``mg_levels_<l>_pc_type
  composite`` and the blocks' LU keys) into the options database and the standard
  paths left them there. A later ``preconditioner = "gamg"`` on an adapt child read the
  composite back on a GAMG level with no member PCs: PETSc error 73. The keys now leave
  the database once the solve has set the FAC levels up, and the warm re-solves keep the
  LU blocks they were set up with.
- **#1017.** ``mesh.deform`` on a mesh that carries a FAC patch kills the patch (Louis's
  ruling, 2026-11-08). The re-solve installs the hierarchy without it and says so in
  ``pc_fallbacks`` and a warning: the fault is absent from the preconditioner until the
  patch is rebuilt. The cover gate does not run on the moved rows.
- **#1012.** The solver instrumentation hung bound methods of its KSP entries on the
  KSPs (monitor, post-solve hook, deadline test) while the entries held the KSPs: a
  cycle through the PETSc object that the collector cannot see. Every discarded SNES
  kept its KSP tree resident, +23 MB per solve when a Stokes ``bodyforce`` is
  reassigned, in serial too. Releasing an entry now takes its callbacks off.

The parallel half of the bundle, #1020 (a reinstall's dropped objects stay on
petsc4py's deferred list), is ``tests/parallel/test_1149_custom_mg_reinstall_frees_parallel.py``.
"""
import gc

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

from test_1146_custom_mg_after_deform import _poisson, _stokes

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _fac_keys_in_database(solver):
    """The FAC smoother keys under this solver's prefix: a level's composite and its
    blocks' sub-solver keys."""
    prefix = solver.petsc_options.prefix or ""
    found = []
    for name, value in PETSc.Options().getAll().items():
        if not name.startswith(prefix) or "mg_levels_" not in name:
            continue
        if "sub_1_sub_" in name or (name.endswith("_pc_type") and value == "composite"):
            found.append(name)
    return found


def _adapt_child_poisson():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(0.3, 0.7, 9), np.linspace(0.45, 0.55, 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface("guide_1149", base, pts)
    guide.discretize()
    mesh = base.adapt(guide.refinement_metric_function(
        h_near=0.025, h_far=0.05, width=0.08, profile="linear"), max_levels=1)
    s = uw.systems.Poisson(mesh)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    return s


def test_gamg_after_a_fac_install_on_an_adapt_child():
    s = _adapt_child_poisson()
    s.solve()
    pc = s.snes.getKSP().getPC()
    assert pc.getType() == "mg"
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() == "composite"
    s.preconditioner = "gamg"
    s.solve()                      # PETSc error 73 before the fix
    assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
    assert s.snes.getKSP().getPC().getType() == "gamg"


def test_fac_keys_leave_the_database_and_the_blocks_keep_lu():
    # Poisson's multigrid is the top-level PC, which sets its levels up inside the
    # solve: the keys must be in the database then, and the warm solves after the drop
    # must keep the blocks' LU (an ASM block left to its default runs ILU(0), #892).
    s, _mesh, managed_pc = _poisson()
    its = []
    for _ in range(3):
        s.solve()
        assert s.snes.getConvergedReason() > 0
        its.append(s.snes.getKSP().getIterationNumber())
        assert _fac_keys_in_database(s) == []
    pc = managed_pc()
    asm = pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getCompositePC(1)
    assert [b.getPC().getType() for b in asm.getASMSubKSP()] == ["lu"] * len(
        asm.getASMSubKSP())
    assert its[0] == its[1] == its[2], its


@pytest.mark.parametrize("build", [_poisson, _stokes], ids=["poisson", "stokes_velocity"])
def test_a_deform_kills_the_fac_patch_and_says_so(build):
    s, mesh, managed_pc = build()
    s.solve()
    pc = managed_pc()
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() == "composite"
    assert "custom_mg.fac_patch" not in s.pc_fallbacks

    c = np.asarray(mesh.X.coords).copy()
    c[:, 0] += 0.01 * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])   # zero on the walls
    mesh.deform(c)
    with pytest.warns(UserWarning, match="FAC patch"):
        s.solve()
    assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
    record = s.pc_fallbacks["custom_mg.fac_patch"]
    assert record["reason"] == "unavailable" and "deform" in record["detail"]
    pc = managed_pc()
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() != "composite"
    # killed, not declined: the cover gate did not run on the moved rows
    assert all(g is None for g in s._custom_mg["hierarchy"].fac_gate)

    # the patch stays killed over later solves and installs
    s.solve()
    pc = managed_pc()
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() != "composite"


def _live_ksps():
    gc.collect()
    return sum(isinstance(o, PETSc.KSP) for o in gc.get_objects())


@pytest.mark.parametrize("guarded", [False, True], ids=["gauge", "guard_armed"])
def test_a_rebuilt_stokes_snes_lets_its_ksps_go(guarded):
    # Reassigning bodyforce rebuilds the SNES on every solve. The old outer KSP and
    # its two fieldsplit blocks stayed alive: +3 KSPs per solve before the fix (H1,
    # #1012). With the guard armed the deadline test held the entry too.
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.25, qdegree=2)
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((0, 0), wall)
    s.petsc_use_pressure_nullspace = True
    if guarded:
        s.guard(wall_per_step=600.0)
    counts = []
    for k in range(5):
        s.bodyforce = sympy.Matrix([[0, (-1) ** k * sympy.sin(sympy.pi * x)]])
        s.solve()
        assert s.snes.getConvergedReason() > 0
        counts.append(_live_ksps())
    assert counts[-1] == counts[1], counts
