"""The composed ribbon FMG benchmark (#629), enshrined.

The 2-D benchmark of record for the fault-patch multigrid architecture
(design note ``docs/developer/design/fault-patch-multigrid-2026-08.md``):
a base ladder (L0, L1), a placed TRANSFINITE ribbon as the unsplit bridge
level, and the finest level = a finer ribbon + spine cut + split +
frictionless contact — PURE CONTACT, no painted weakness anywhere: the
fault is the split, the ribbon is resolution.

The patch-keying ruling this file encodes (Louis, 2026-08-24): a
split-node fault runs the STRUCTURAL patch alone — the split exists to
be efficient, and its patch (split pairs + cut-inserted DOFs) is
trace-proportional and installed automatically. The ``fac_zone`` key is
for VOLUMETRIC fault representations (weak / TI / damage rheology,
where the zone width is physics); the ribbon itself is never the key.

What must hold, each measured on the campaign rig and each the
signature of a specific failure when violated:

* **Native-like level densities** (nnz/row bounded): structural-zero
  weights in the transfers fatten the Galerkin chain 3–5x per level.
* **A strong finest ASM patch**: the patch smoother replaces whole-level
  smoothing, so a finest level missing the cut/split rows leaves them
  smoothed nowhere and the velocity sub-solve caps on every application.
* **Few velocity/pressure iterations** (recorded: vel 8–9, pres 13
  against GAMG 82): a silently declined patch stalls or caps.
* **Slip/leak invariants** (recorded: slip 0.2223, leak ~2e-17): the
  physics must be indifferent to the preconditioner.

Serial: the composed placement + split pipeline is serial-only (the
parallel split is #629 phase-2 work).
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import fault_contact
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_b,
    pytest.mark.skipif(uw.mpi.size > 1,
                       reason="composed split pipeline is serial"),
]

H_BG = 1.0 / 8          # L0 cell size (refinement=1 halves it for L1)
WIDTH = 0.06            # ribbon width
S_MID = 0.020           # band spacing, intermediate (UNSPLIT) level
S_FINE = 0.010          # band spacing, finest level
TIP_RUNGS = 3           # cut stops this many rungs inside the band
TRACE = np.array([[0.15, 0.42], [0.85, 0.58]])
FAULT = "Main"


def _banded(base, size):
    """The placed transfinite ribbon at the given spacing (#595 ladder)."""
    for clearance in (0.55, 0.7, 0.9):
        try:
            dm, _info = place_thin_volume(
                base.dm, [TRACE], width=WIDTH, label="Zone", label_value=31,
                size=size, clearance=clearance, assembly="fuse",
                mesher="ladder")
            return uw.discretisation.Mesh(
                dm, simplex=True, qdegree=2, boundaries=base.boundaries,
                coordinate_system_type=base.CoordinateSystem.coordinate_type)
        except RuntimeError:
            continue
    raise RuntimeError(f"band build failed at size {size}")


@pytest.fixture(scope="module")
def composed_stack():
    """base (L0+L1) -> mid ribbon (unsplit) -> finest ribbon + cut + split."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=H_BG,
        regular=False, qdegree=2, refinement=1)
    mid = _banded(base, S_MID)
    fine0 = _banded(base, S_FINE)

    # The spine cut, sampled AT the transfinite centreline's own vertices.
    a, b = TRACE[0], TRACE[-1]
    t = (b - a) / np.linalg.norm(b - a)
    L = float(np.linalg.norm(b - a))
    n_along = max(2, int(round(L / S_FINE)))
    line = a + (np.arange(TIP_RUNGS, n_along - TIP_RUNGS + 1)
                * (L / n_along))[:, None] * t
    mesh = fine0.add_fault((FAULT, line))
    tail = base._coarse_level_meshes()[:-1] + [base, mid]
    return mesh, tail


def _contact_stokes(mesh, tag):
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1,
                                       continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, boundary=FAULT)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5
    stokes.strategy = "robust"
    return stokes


def _solve_and_probe(stokes):
    info = fault_contact.solve_with_fault(stokes)
    assert info.get("converged")
    ctx = (getattr(stokes, "_rotated_linear_cache", None) or {})["ctx"]
    velpc = ctx["ksp"].getPC().getFieldSplitSubKSP()[0].getPC()
    assert velpc.getType() == "mg", "custom-P PCMG was not installed"
    return ctx, velpc


def _patch_pc(spc):
    """The ASM patch of a level: the level PC itself in the replacing form,
    the second member of the composite in the default form (the whole-level
    smoother first, the fault blocks on top — #670)."""
    if spc.getType() == "composite":
        return spc.getCompositePC(1)
    return spc


def test_pure_contact_structural_patch(composed_stack):
    """The benchmark of record: split contact, structural patch alone."""
    mesh, tail = composed_stack
    stokes = _contact_stokes(mesh, "A")
    set_custom_fmg(stokes, tail, field_id=0)          # fac_zone=None
    ctx, velpc = _solve_and_probe(stokes)

    vel_its, pres_its = ctx.get("vel_its_last"), ctx.get("pres_its_last")
    # Recorded 8-9 / 13. A violated structural-patch rule caps at 200;
    # GAMG-class behaviour is 82; the bounds separate healthy from broken.
    assert vel_its is not None and vel_its <= 15, f"velocity: {vel_its} its"
    assert pres_its is not None and pres_its <= 40, f"pressure: {pres_its} its"

    nlev = velpc.getMGLevels()
    assert nlev == len(tail) + 1
    finest_blocks = 0
    for l in range(nlev):
        sm = velpc.getMGSmoother(l)
        A = sm.getOperators()[0]
        nnz_row = A.getInfo()["nz_used"] / max(A.getSize()[0], 1)
        # Recorded 24-28 everywhere; structural-zero fattening measured
        # 90-481. The bound is the fat-chain tripwire, not a target.
        assert nnz_row <= 40, f"level {l}: nnz/row {nnz_row:.0f} (fat chain)"
        spc = _patch_pc(sm.getPC())
        if spc.getType() == "asm":
            assert not spc.getFailedReason(), f"level {l} ASM PC failed"
            if l == nlev - 1:
                finest_blocks = len(spc.getASMSubKSP())
    assert finest_blocks >= 1, (
        "no finest ASM patch: the structural (split) rows are smoothed "
        "nowhere and only survive by luck of the smoother")

    _c, jumps, normals = fault_contact.fault_pair_jumps(
        stokes, FAULT, stokes._rotated_freeslip_info)
    jn = np.einsum("ij,ij->i", jumps, normals)
    slip = float(np.linalg.norm(jumps - jn[:, None] * normals, axis=1).max())
    leak = float(np.abs(jn).max())
    assert leak < 1e-10, f"contact leak {leak:.2e}"
    assert abs(slip - 0.2223) < 0.01, f"peak slip {slip:.4f} (recorded 0.2223)"


def test_fac_zone_key_installs_union_block(composed_stack):
    """The volumetric-zone API: a fac_zone mask installs its own block AND
    the automatic structural union (the uncovered cut/split rows). Keyed
    here on the ribbon label purely to exercise the machinery — in a real
    model the key is a painted rheological zone, never the ribbon."""
    mesh, tail = composed_stack
    zone = mesh.cells_labelled("Zone", 31)
    assert zone.any(), "the ribbon's Zone label must survive the cut/split"

    stokes = _contact_stokes(mesh, "B")
    set_custom_fmg(stokes, tail, field_id=0, fac_zone=zone)
    ctx, velpc = _solve_and_probe(stokes)
    assert ctx.get("vel_its_last") <= 15

    spc = _patch_pc(velpc.getMGSmoother(velpc.getMGLevels() - 1).getPC())
    assert spc.getType() == "asm"
    level_pc = velpc.getMGSmoother(velpc.getMGLevels() - 1).getPC()
    if level_pc.getType() == "composite":
        # the default form (#670): the whole-level smoother covers the
        # structural rows, and the zone block stands alone on top of it
        assert len(spc.getASMSubKSP()) >= 1, "zone key must install the zone block"
    else:
        assert len(spc.getASMSubKSP()) >= 2, (
            "zone key must install the zone block plus the automatic "
            "structural-union block")

    # The retired spelling must fail LOUDLY, never decline silently.
    stokes2 = _contact_stokes(mesh, "C")
    set_custom_fmg(stokes2, tail, field_id=0)
    stokes2._fac_zone_cells = zone
    with pytest.raises(RuntimeError, match="fac_zone"):
        fault_contact.solve_with_fault(stokes2)
