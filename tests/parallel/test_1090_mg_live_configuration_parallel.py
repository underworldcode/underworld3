"""Every multigrid level runs the smoother it was configured with (#892).

The FAC patch smoother (#629, #670) solves the fault band's blocks by LU with a
nonzero factor shift: ``custom_mg._configure_pcmg`` writes
``mg_levels_<l>_sub_1_sub_pc_type lu`` and its shift into the options database,
and the block KSPs read those keys when the patch's ASM sets up. On the rotated
free-slip path the keys are deleted right after the multigrid set-up, but in the
default (composite) form the ASM is the composite's second member, which sets
up at its first application, after the deletion. The blocks then took PCASM's
default, unshifted ILU(0), on both rotated call sites (the saddle-point solve
and the thin sheet), from 891c9bb0 until the fix. Measured on the linear fault
fixture of UWTN 2026-022 at band ratio 24: 344 velocity iterations with the ILU
blocks, 67 with LU.

An options-database check cannot see this, because the keys were right all
along: the failure is a key deleted before the object that reads it exists. So
this test reads the LIVE objects after the solve, on every path that installs
the custom multigrid:

* ``stokes_rotated``: frictionless fault contact, the rotated saddle-point solve;
* ``sheet_rotated``: the same contact on a thin sheet, the rotated vector solve;
* ``stokes_plain``: the same split mesh with no fault BC (a traction-free
  crack), the velocity-block install of the standard solve;
* ``poisson``: the top-level PC install of a scalar solve.

each in the composite (default) and plain ASM forms (``UW_FAC_MODE``), and checks
every level against the configuration ``_configure_pcmg`` and the "robust"
geometric bundle write: coarse ``preonly`` + ``redundant`` with LU inside; every
other level ``gmres``, 4 iterations, no norm, SOR; the FAC level a composite of
SOR then ASM (or the ASM alone), basic restriction, one overlap layer, every
block ``preonly`` + LU with the nonzero shift.

The fixture is a fault split in place over a welded base hierarchy (1/8, 1/16),
so the finest level is the split mesh and the structural patch puts the FAC
smoother on it in serial and in parallel. A second, serial test pins which levels
carry a patch on test_1022's composed stack, where a coarse level has one and the
rotated and standard installs disagree on keeping it. This file carries no
``mpi(min_size=2)`` mark: it runs in the serial suite and in the parallel batches.

    python -m pytest tests/parallel/test_1090_mg_live_configuration_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1090_mg_live_configuration_parallel.py
"""
import os
import tempfile

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])
PATHS = ("stokes_rotated", "sheet_rotated", "stokes_plain", "poisson")


@pytest.fixture(scope="module")
def base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=1)


def _solved_mg_pc(base, path):
    """Solve on the split mesh by ``path``; return the live multigrid PC."""
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = base.add_fault(("Flt", TRACE))
    x, y = mesh.X
    if path.startswith("stokes"):
        v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
        solver = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        for wall in ("Bottom", "Top", "Left", "Right"):
            solver.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
        if path == "stokes_rotated":
            solver.add_fault_bc(0, "Flt")
        solver.petsc_use_pressure_nullspace = True
    elif path == "sheet_rotated":
        solver = uw.systems.ThinSheet(mesh)
        for wall in ("Bottom", "Top", "Left", "Right"):
            solver.add_essential_bc([y, 0], wall)
        solver.add_fault_bc(0, "Flt")
    else:
        solver = uw.systems.Poisson(mesh)
        solver.f = 1
        for wall in ("Bottom", "Top", "Left", "Right"):
            solver.add_dirichlet_bc(0, wall)
    if path == "poisson":
        solver.constitutive_model = uw.constitutive_models.DiffusionModel
    else:
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        solver.constitutive_model.Parameters.shear_viscosity_0 = 1
    solver.tolerance = 1e-6
    custom_mg.set_custom_fmg(solver, levels, field_id=0 if path.startswith("stokes") else None)
    solver.solve()

    if path.endswith("rotated"):
        info = solver._rotated_freeslip_info
        assert info["converged"], f"{path}: the rotated solve did not converge"
        ctx = solver._rotated_linear_cache["ctx"]
        if path == "sheet_rotated":
            return ctx["pc"]
        return ctx["ksp"].getPC().getFieldSplitSubKSP()[0].getPC()
    assert solver.snes.getConvergedReason() > 0, f"{path}: the solve did not converge"
    pc = solver.snes.getKSP().getPC()
    return pc.getFieldSplitSubKSP()[0].getPC() if path == "stokes_plain" else pc


def _view(obj):
    """``obj.view()`` as text, the same on every rank of the object's communicator.
    petsc4py has no getter for a factor PC's shift, a redundant PC's inner solver
    or an ASM's overlap and restriction type; the view reports all three."""
    comm = obj.getComm().tompi4py()
    path = None
    if comm.rank == 0:
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
    path = comm.bcast(path)
    viewer = PETSc.Viewer().createASCII(path, comm=obj.getComm())
    obj.view(viewer)
    viewer.destroy()
    text = None
    if comm.rank == 0:
        with open(path) as f:
            text = f.read()
        os.remove(path)
    return comm.bcast(text)


def _checked_fac_levels(mg, path, mode):
    """Check every level of the live PCMG ``mg`` against its configuration and
    return the levels that carry a FAC patch."""
    assert mg.getType() == "mg", f"{path}: custom multigrid not installed"
    nlev = mg.getMGLevels()
    coarse = mg.getMGSmoother(0)
    assert coarse.getType() == "preonly"
    assert coarse.getPC().getType() == "redundant"
    inner = _view(coarse.getPC()).split("Redundant preconditioner", 1)[-1]
    assert "type: lu" in inner, f"{path}: the coarse solve is not redundant LU:\n{inner}"

    fac_levels = []
    for l in range(1, nlev):
        sm = mg.getMGSmoother(l)
        assert sm.getType() == "gmres", f"{path} level {l}: smoother KSP {sm.getType()}"
        assert sm.getTolerances()[3] == 4, f"{path} level {l}: {sm.getTolerances()[3]} its"
        assert sm.getNormType() == PETSc.KSP.NormType.NONE
        pc = sm.getPC()
        if pc.getType() == "sor":
            continue
        fac_levels.append(l)
        if mode == "composite":
            assert pc.getType() == "composite", f"{path} level {l}: {pc.getType()}"
            assert [pc.getCompositePC(i).getType() for i in range(2)] == ["sor", "asm"]
            asm = pc.getCompositePC(1)
        else:
            assert pc.getType() == "asm", f"{path} level {l}: {pc.getType()}"
            asm = pc
        asm_text = _view(asm)
        assert "amount of overlap = 1" in asm_text
        assert "restriction/interpolation type - BASIC" in asm_text
        blocks = asm.getASMSubKSP()
        for k in blocks:
            assert k.getType() == "preonly"
            # the #892 failure: unshifted ILU(0) here instead of the shifted LU
            assert k.getPC().getType() == "lu", (
                f"{path} ({mode}) level {l}: FAC block solver is "
                f"{k.getPC().getType()!r}, configured 'lu' (#892)")
            assert "[NONZERO]" in _view(k.getPC()), (
                f"{path} ({mode}) level {l}: FAC block LU without the nonzero shift")
    return fac_levels


@pytest.mark.parametrize("mode", ["composite", "asm"])
@pytest.mark.parametrize("path", PATHS)
def test_every_level_runs_its_configured_smoother(base, path, mode, monkeypatch):
    monkeypatch.setenv("UW_FAC_MODE", mode)
    mg = _solved_mg_pc(base, path)
    assert mg.getMGLevels() == 3
    # the split mesh is the finest level, so the structural patch is always there
    # (the coarse pair is a native refinement and takes none): without it this
    # test would pass on any solver that never installs FAC
    assert _checked_fac_levels(mg, path, mode) == [2], f"{path}: FAC levels"


H_BASE, WIDTH, S_MID, S_FINE = 1 / 8, 0.06, 0.020, 0.010
RIBBON = np.array([[0.15, 0.42], [0.85, 0.58]])


def _ribbon(base, size):
    """test_1022's placed transfinite ribbon at band spacing ``size``."""
    from underworld3.utilities.place_surface import place_thin_volume
    for clearance in (0.55, 0.7, 0.9):
        try:
            dm, _info = place_thin_volume(
                base.dm, [RIBBON], width=WIDTH, label="Zone", label_value=31,
                size=size, clearance=clearance, assembly="fuse", mesher="ladder")
        except RuntimeError:
            continue
        return uw.discretisation.Mesh(
            dm, simplex=True, qdegree=2, boundaries=base.boundaries,
            coordinate_system_type=base.CoordinateSystem.coordinate_type)
    raise RuntimeError(f"ribbon placement failed at size {size}")


@pytest.mark.skipif(uw.mpi.size > 1, reason="the composed placement is serial")
@pytest.mark.parametrize("path,expected", [("stokes_rotated", [3]),
                                           ("stokes_plain", [2, 3])])
def test_which_levels_carry_a_fac_patch(path, expected, monkeypatch):
    """test_1022's composed stack (base L0 + L1, an unsplit mid ribbon, the finest
    ribbon cut and split) gives a coarse level a patch: the mid ribbon is not a
    refinement of the base, so its transfer has rows a SOR level cannot smooth.

    The two installs disagree on keeping it, and this pins both. The standard
    install keeps the patch rows of every level the orphan restriction leaves
    whole (``CustomMGHierarchy.install``); the rotated path clears every coarse
    level's rows after the restriction and keeps only the finest's
    (``rotated_bc``, the ``drop_orphans`` block), so the contact solve smooths the
    mid ribbon by whole-level SOR. Measured 2026-10-30 at 799ef021 and with the
    #892 fix; the rule is B's (#889) and not yet ruled on. A change to either
    install must change this test deliberately."""
    monkeypatch.setenv("UW_FAC_MODE", "composite")
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=H_BASE, regular=False,
        qdegree=2, refinement=1)
    mid = _ribbon(base, S_MID)
    a, b = RIBBON
    length = float(np.linalg.norm(b - a))
    n_along = int(round(length / S_FINE))
    spine = a + (np.arange(3, n_along - 2) * (length / n_along))[:, None] * (b - a) / length
    mesh = _ribbon(base, S_FINE).add_fault(("Flt", spine))
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - sympy.Rational(1, 2), 0), wall)
    if path == "stokes_rotated":
        stokes.add_fault_bc(0, "Flt")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    custom_mg.set_custom_fmg(stokes, base._coarse_level_meshes()[:-1] + [base, mid],
                             field_id=0)
    stokes.solve()
    if path == "stokes_rotated":
        assert stokes._rotated_freeslip_info["converged"]
        mg = (stokes._rotated_linear_cache["ctx"]["ksp"].getPC()
              .getFieldSplitSubKSP()[0].getPC())
    else:
        assert stokes.snes.getConvergedReason() > 0
        mg = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC()
    assert mg.getMGLevels() == 4
    assert _checked_fac_levels(mg, path, "composite") == expected
