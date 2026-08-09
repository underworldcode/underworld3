"""Parallel 3D ``mesh.adapt()`` — stage 1c-iii of the adaptivity capstone.

The native driver refines tetrahedral meshes in place, co-partitioned with
the parent, using the same cross-rank machinery as the 2D engine (requested
edges exchanged over the point SF; an edge splits only when every cell
around it — on any rank — nominates it; a shared edge chosen anywhere splits
everywhere). The refinement state is seeded identically on every rank from
geometry alone (``write_tagged_state_label``), so the refined mesh is
**partition-independent**: the asserts below hold serially and under
``mpirun -np K`` unchanged — the 3D mirror of ``test_0839``.

Runs on 1..N ranks; skips cleanly where the native transform is not built.
"""
import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

pytest.importorskip(
    "underworld3.utilities._nvb_transform",
    reason="native uwnvb transform not built (needs the custom-PETSc/amr env)",
)


def _base3():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), cellSize=0.4,
        refinement=1, qdegree=2)


def _ball_metric(centroids):
    """Callable metric M = 1/h(r)^2 — evaluated per level at each rank's own
    centroids (coordinate-driven, hence partition-independent)."""
    r = np.linalg.norm(np.asarray(centroids) - 0.5, axis=1)
    h = np.where(r < 0.18, 0.06,
                 np.minimum(0.06 + (0.4 - 0.06) * (r - 0.18) / 0.25, 0.4))
    return 1.0 / h**2


def _global_owned_cells(dm):
    from mpi4py import MPI

    cs, ce = dm.getHeightStratum(0)
    sf = dm.getPointSF()
    try:
        _, ilocal, _ = sf.getGraph()
    except (ValueError, TypeError):
        ilocal = None
    leaves = set() if ilocal is None else {int(x) for x in ilocal}
    owned = sum(1 for c in range(cs, ce) if c not in leaves)
    return dm.comm.tompi4py().allreduce(owned, op=MPI.SUM)


def test_3d_adapt_is_confluent_and_carries_labels():
    """The graded 3D child is partition-independent and keeps its labels."""
    mesh = _base3()
    parent_cells = _global_owned_cells(mesh.dm)
    child = mesh.adapt(_ball_metric, max_levels=1)      # engine-less default
    child_cells = _global_owned_cells(child.dm)
    uw.pprint(f"3D adapt gate: base {parent_cells} cells -> child {child_cells} cells")

    assert child_cells > parent_cells

    # Confluence: the refined count must not depend on the communicator
    # size — that is the partition-independence claim, and it is pinned
    # per-environment. The ABSOLUTE count is NOT environment-portable:
    # CI (conda PETSc 3.25.3) refines the SAME 1472-cell base to 4816
    # cells where the reference toolchain (PETSc 3.25.0) gives 5198 —
    # threshold marking is FP-sensitive and the conformity closure
    # amplifies the flips (tracked as a cross-version determinism
    # question; the structural gates above and the MG test below pass on
    # both). Pin the count only in the reference toolchain; the logged
    # counts keep drift visible everywhere else.
    import petsc4py

    if parent_cells == 1472 and petsc4py.__version__ == "3.25.0":
        assert child_cells == 5198

    assert child.parent is mesh
    assert child._relationship_kind == "refinement"
    names = {b.name for b in child.boundaries}
    assert {"Top", "Bottom", "Left", "Right", "Front", "Back"} <= names
    assert len(child._custom_mg_coarse_meshes) >= 2


def test_poisson_fmg_on_3d_child_matches_gamg():
    """Custom-P geometric MG on the graded 3D child converges like GAMG,
    at any communicator size."""
    mesh = _base3()
    child = mesh.adapt(_ball_metric, max_levels=1)

    # Deliberate ordering: create BOTH variables before any solver runs —
    # creating a MeshVariable after a solve rebuilds mesh.dm and detonates
    # issue #492 (the old DM is destroyed under the custom-MG coarse/fine
    # links; that dangling reference is what segfaulted Linux CI downstream).
    fields = {pc: uw.discretisation.MeshVariable(f"u_{pc}", child, 1, degree=1)
              for pc in ("fmg", "gamg")}

    def solve(pc):
        u = fields[pc]
        poisson = uw.systems.Poisson(child, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = 0.0
        poisson.add_dirichlet_bc(0.0, "Bottom")         # z = 0
        poisson.add_dirichlet_bc(1.0, "Top")            # z = 1
        if pc == "gamg":
            poisson.preconditioner = "gamg"
            poisson.petsc_options["pc_type"] = "gamg"
        poisson.petsc_options["ksp_rtol"] = 1e-9
        poisson.solve()
        ksp = poisson.snes.getKSP()
        its = ksp.getIterationNumber()
        # The comparison must be REAL. The explicit-gamg arm was once silently
        # clobbered by the mesh-owned custom-P pickup, so both arms ran
        # pc_type=mg and the fmg-vs-gamg comparison compared FMG to itself.
        # Pin each arm's PC so that vacuous comparison can never return.
        assert ksp.getPC().getType() == ("gamg" if pc == "gamg" else "mg")
        # exact linear solution T = z: also proves the Dirichlet facet
        # labels survived the parallel transform.
        #
        # The bound is tight ON PURPOSE: it must catch a once-shipped defect
        # whose signature was a TRUE-error stall at 1e-6, INSENSITIVE to
        # ksp_rtol — the gmres-smoothed geometric bundle (a non-stationary
        # preconditioner) under a plain left-preconditioned gmres outer, whose
        # recurrence norm fell to 1e-11 while the true residual stalled
        # (custom_mg._ensure_flexible_outer now pairs that bundle with fgmres,
        # so the fmg arm's ksp_rtol is enforced in the true residual norm;
        # measured err/nrm ~1e-12). The gamg arm still converges in the
        # preconditioned norm, with a declared-reduction -> nodal-error
        # constant of ~10 on this child, so the declared reduction is one
        # order tighter than the bound: neither arm rides on its PC constant,
        # and the O(1) failures this test exists for (a lost Dirichlet label,
        # a wrong transfer) stay unmissable.
        err = np.linalg.norm(
            poisson.Unknowns.u.data[:, 0] - poisson.Unknowns.u.coords[:, 2])
        nrm = np.linalg.norm(poisson.Unknowns.u.coords[:, 2]) + 1e-30
        assert err / nrm < 1e-7
        return its

    fmg_its = solve("fmg")
    gamg_its = solve("gamg")
    assert fmg_its <= gamg_its + 2, f"fmg {fmg_its} vs gamg {gamg_its}"
    assert fmg_its <= 12
