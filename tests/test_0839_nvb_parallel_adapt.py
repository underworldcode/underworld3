"""Layer-2 Route B, Stage 2c: parallel ``mesh.adapt(engine="nvb")``.

The native ``uwnvb`` ``DMPlexTransform`` is the parallel NVB engine — in-place
(co-partitioned with the parent, so the custom-P geometric-MG tail stays valid),
graded, and bit-confluent serial↔parallel. ``discretisation_mesh._adapt_nested``
dispatches to it whenever the compiled extension is present.

These tests pin the integrated parallel path:
  - **confluence**: the adapted child's global (owner-counted) cell total is
    partition-independent — identical at any communicator size, so the same
    assert holds serially and under ``mpirun``;
  - **labels carried**: the child keeps the parent's boundary labels through the
    transform, and owns the ``[base … child]`` custom-P MG tail;
  - **FMG acceptance**: a Poisson solve on the graded child drives the mesh-owned
    custom-P geometric multigrid (``preconditioner="auto"``) and converges in the
    same iteration count as a GAMG reference.

Runs on 1..N ranks; the asserts are partition-independent, so ``mpirun -np K``
validates confluence directly.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw
from petsc4py import PETSc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# Parallel NVB needs the native transform; skip cleanly where it is not built.
pytest.importorskip(
    "underworld3.utilities._nvb_transform",
    reason="native uwnvb transform not built (needs the custom-PETSc/amr env)",
)


def _base():
    """A base mesh with a one-level MG tail (refinement=1) so nested adapt has a
    coarse hierarchy to extend."""
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=False,
        qdegree=2, refinement=1,
    )


def _bullseye_metric(mesh, h_near=0.05, h_far=0.4, radius=0.25):
    M = uw.discretisation.MeshVariable("Mbe", mesh, 1, degree=1)
    with mesh.access(M):
        c = M.coords
        d = np.sqrt((c[:, 0] - 0.5) ** 2 + (c[:, 1] - 0.5) ** 2)
        h = np.where(d < radius, h_near, h_far)
        M.data[:, 0] = 1.0 / h ** 2
    return M


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


def test_nvb_adapt_is_confluent_and_carries_labels():
    """The graded child is partition-independent and keeps its lineage/labels."""
    mesh = _base()
    child = mesh.adapt(_bullseye_metric(mesh), max_levels=2, engine="nvb")

    # confluence: same global cell count at any communicator size (serial value)
    assert _global_owned_cells(child.dm) == 366

    # lineage + boundary labels carried through the transform
    assert child.parent is mesh
    assert child._relationship_kind == "refinement"
    names = {b.name for b in child.boundaries}
    assert {"Top", "Bottom", "Left", "Right"} <= names

    # mesh-owned custom-P tail: [base L0, base finest, nvb-1, nvb-2] under child
    assert len(child._custom_mg_coarse_meshes) >= 2


def test_poisson_fmg_on_nvb_child_matches_gamg():
    """Custom-P geometric MG on the graded NVB child converges like GAMG."""
    mesh = _base()
    child = mesh.adapt(_bullseye_metric(mesh), max_levels=2, engine="nvb")

    x, y = child.X
    exact = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)

    def solve(pc):
        u = uw.discretisation.MeshVariable(f"u_{pc}", child, 1, degree=1)
        poisson = uw.systems.Poisson(child, u_Field=u)
        poisson.constitutive_model = uw.constitutive_models.DiffusionModel
        poisson.constitutive_model.Parameters.diffusivity = 1.0
        poisson.f = 2 * sympy.pi ** 2 * exact
        for b in ("Top", "Bottom", "Left", "Right"):
            poisson.add_dirichlet_bc(0.0, b)
        poisson.preconditioner = "auto" if pc == "fmg" else "gamg"
        if pc == "gamg":
            poisson.petsc_options["pc_type"] = "gamg"
        poisson.petsc_options["ksp_rtol"] = 1e-8
        poisson.solve()
        return poisson.snes.getKSP().getIterationNumber()

    fmg_its = solve("fmg")
    gamg_its = solve("gamg")
    # both are effective preconditioners on this small graded mesh; the custom-P
    # FMG must not need materially more iterations than the AMG reference.
    assert fmg_its <= gamg_its + 2, f"fmg {fmg_its} vs gamg {gamg_its}"
    assert fmg_its <= 10
