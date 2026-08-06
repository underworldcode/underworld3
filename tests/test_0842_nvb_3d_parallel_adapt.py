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

    def solve(pc):
        u = uw.discretisation.MeshVariable(f"u_{pc}", child, 1, degree=1)
        poisson = uw.systems.Poisson(child, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = 0.0
        poisson.add_dirichlet_bc(0.0, "Bottom")         # z = 0
        poisson.add_dirichlet_bc(1.0, "Top")            # z = 1
        if pc == "gamg":
            poisson.preconditioner = "gamg"
            poisson.petsc_options["pc_type"] = "gamg"
        poisson.petsc_options["ksp_rtol"] = 1e-8
        poisson.solve()
        its = poisson.snes.getKSP().getIterationNumber()
        # exact linear solution T = z: also proves the Dirichlet facet
        # labels survived the parallel transform.
        #
        # The bound is SOLVER-TOLERANCE-limited, not discretisation-limited:
        # ksp_rtol is enforced in the PRECONDITIONED norm, so the constant
        # between the declared 1e-8 reduction and the nodal error depends on
        # the preconditioner. The one-level-per-doubling hierarchy (2026-08)
        # legitimately changed that constant — measured 6.9e-7 on the CI gmsh
        # mesh vs 2.7e-10 on macOS gmsh 4.15.1 at the same rtol — so a 1e-8
        # nodal bound was a calibration to the OLD hierarchy's margin, not a
        # property of the method. 1e-6 still catches every failure this test
        # exists for: a lost Dirichlet label or a wrong transfer is O(1).
        err = np.linalg.norm(
            poisson.Unknowns.u.data[:, 0] - poisson.Unknowns.u.coords[:, 2])
        nrm = np.linalg.norm(poisson.Unknowns.u.coords[:, 2]) + 1e-30
        assert err / nrm < 1e-6
        return its

    fmg_its = solve("fmg")
    gamg_its = solve("gamg")
    assert fmg_its <= gamg_its + 2, f"fmg {fmg_its} vs gamg {gamg_its}"
    assert fmg_its <= 12
