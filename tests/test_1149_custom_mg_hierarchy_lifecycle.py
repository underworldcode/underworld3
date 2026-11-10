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


def _adapt_child(name):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(0.3, 0.7, 9), np.linspace(0.45, 0.55, 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface(name, base, pts)
    guide.discretize()
    return base.adapt(guide.refinement_metric_function(
        h_near=0.025, h_far=0.05, width=0.08, profile="linear"), max_levels=1)


def _adapt_child_poisson():
    s = uw.systems.Poisson(_adapt_child("guide_1149"))
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.f = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc(0, wall)
    s.tolerance = 1.0e-6
    return s, lambda: s.snes.getKSP().getPC()


def _adapt_child_stokes():
    mesh = _adapt_child("guide_1149s")
    _x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-6
    return s, lambda: s.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC()


@pytest.mark.parametrize("build", [_adapt_child_poisson, _adapt_child_stokes],
                         ids=["poisson", "stokes_velocity"])
def test_gamg_after_a_fac_install_on_an_adapt_child(build):
    # Poisson's keys are dropped after the solve, the velocity block's at its
    # explicit set-up; either left behind is error 73 on the GAMG levels
    s, managed_pc = build()
    s.solve()
    pc = managed_pc()
    assert pc.getType() == "mg"
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() == "composite"
    s.preconditioner = "gamg"
    s.solve()                      # PETSc error 73 before the fix
    assert s.snes.getConvergedReason() > 0, s.snes.getConvergedReason()
    assert managed_pc().getType() == "gamg"


def test_a_user_block_ordering_survives_the_drop():
    # The install writes qmd only where the database has no ordering, and drops
    # only what it wrote: a user's ordering stays for the next set-up.
    s, _mesh, managed_pc = _poisson()
    fac = len(s._custom_mg["hierarchy"].level_meshes) - 1      # the finest level
    key = f"mg_levels_{fac}_sub_1_sub_pc_factor_mat_ordering_type"
    s.petsc_options[key] = "rcm"
    s.solve()
    assert s.snes.getConvergedReason() > 0
    prefix = s.petsc_options.prefix or ""
    assert PETSc.Options().getString(prefix + key, None) == "rcm"
    assert _fac_keys_in_database(s) == [prefix + key]        # the install's are gone


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
    # killed, not declined: the cover gate did not run on the moved rows
    _killed_and_said_so(s)
    pc = managed_pc()
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() != "composite"

    # the patch stays killed over later solves and installs
    s.solve()
    pc = managed_pc()
    assert pc.getMGSmoother(pc.getMGLevels() - 1).getPC().getType() != "composite"


def _killed_and_said_so(s):
    record = s.pc_fallbacks["custom_mg.fac_patch"]
    assert record["reason"] == "unavailable" and "deform" in record["detail"]
    assert all(g is None for g in s._custom_mg["hierarchy"].fac_gate)


def test_a_deform_kills_the_patch_on_the_rotated_fault_route():
    # add_fault_bc configures the velocity block in utilities.rotated_bc, not through
    # install(); the route recorded no geometry, and a deform there declined the moved
    # patch silently at the cover gate (H1's review of bundle A, M1)
    s, mesh, _managed_pc = _stokes()
    s.add_fault_bc(0, "Flt")
    s.solve()
    assert s._rotated_freeslip_info["converged"]
    assert "custom_mg.fac_patch" not in s.pc_fallbacks
    assert any(g for g in s._custom_mg["hierarchy"].fac_gate if g is not None)

    c = np.asarray(mesh.X.coords).copy()
    c[:, 0] += 0.01 * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    mesh.deform(c)
    with pytest.warns(UserWarning, match="FAC patch"):
        s.solve()
    assert s._rotated_freeslip_info["converged"]
    _killed_and_said_so(s)


def test_a_mover_kills_the_patch_too():
    # A mover commits through remesh_with_field_transfer as mesh.deform does, and
    # only mesh.deform moved _mesh_version: a smoothed mesh kept a stale patch
    # version and the moved patch was declined silently (H1's review, M2)
    s, mesh, managed_pc = _poisson()
    s.solve()
    before = mesh._mesh_version
    uw.meshing.smooth_mesh_interior(mesh, n_iters=2)
    assert mesh._mesh_version == before + 1
    with pytest.warns(UserWarning, match="FAC patch"):
        s.solve()
    assert s.snes.getConvergedReason() > 0
    _killed_and_said_so(s)


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
