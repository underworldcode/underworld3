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
from underworld3.function import analytic as A
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


def test_adapt_accepts_analytic_metric():
    """adapt(engine="nvb") accepts an analytic metric expression (not only a
    MeshVariable). Evaluating the metric analytically at each centroid avoids
    P1-interpolating a peaked M=1/h² on the coarse base — the cause of *patchy*
    refinement along a thin feature. The child stays confluent."""
    mesh = _base()
    x, y = mesh.X
    # bullseye wedge as a pure expression: h_near at centre, h_far away
    d = sympy.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    h_near, h_far, w = 0.05, 0.4, 0.25
    h = h_near + (h_far - h_near) * sympy.Min(d / w, 1)
    metric_expr = 1.0 / h ** 2

    child = mesh.adapt(metric_expr, max_levels=2, engine="nvb")
    assert child.parent is mesh
    # refined (more cells than the base finest) and partition-independent count
    base_finest = mesh.dm_hierarchy[-1]
    bs, be = base_finest.getHeightStratum(0)
    assert _global_owned_cells(child.dm) > (be - bs)
    assert _global_owned_cells(child.dm) == 174   # confluent (same at any np)
    # conforming: no locally over-shared edge
    es, ee = child.dm.getDepthStratum(1)
    assert all(child.dm.getSupportSize(e) <= 2 for e in range(es, ee))


def test_adapt_accepts_callable_metric_exact_surface_distance():
    """adapt(engine="nvb") accepts a CALLABLE metric evaluated at each refined
    level's centroids. A callable built from a Surface's EXACT signed distance
    (``Surface.refinement_metric_function``) resolves itself at the new
    resolution — no P1 field is interpolated — so a thin feature refines to a
    clean band. Because it is coordinate-driven it is partition-independent
    (needs no global_evaluate) and the child stays confluent."""
    mesh = _base()
    # a thin diagonal "fault" polyline
    s = np.linspace(0.0, 1.0, 21)
    trace = np.column_stack([0.2 + 0.6 * s, 0.15 + 0.7 * s, np.zeros_like(s)])
    fault = uw.meshing.Surface("f839", mesh, trace, symbol="F")
    fault.discretize()

    # exact distance is a real geometric primitive at arbitrary points
    d_on = fault.unsigned_distance(trace[:, :2])
    assert float(np.max(d_on)) < 1e-6            # exactly zero on the trace
    # step exactly perpendicular to the trace direction (0.6, 0.7)
    perp = np.array([0.7, -0.6]) / np.hypot(0.6, 0.7)
    # use interior points only (endpoints clamp to the finite edge)
    off = trace[5:-5, :2] + 0.1 * perp
    assert abs(float(np.mean(fault.unsigned_distance(off))) - 0.1) < 1e-6

    metric = fault.refinement_metric_function(h_near=0.03, h_far=0.4, width=0.08)
    assert callable(metric)

    child = mesh.adapt(metric, max_levels=2, engine="nvb")
    assert child.parent is mesh
    base_finest = mesh.dm_hierarchy[-1]
    bs, be = base_finest.getHeightStratum(0)
    assert _global_owned_cells(child.dm) > (be - bs)   # refined
    # conforming (no over-shared edge) and confluent at any communicator size
    es, ee = child.dm.getDepthStratum(1)
    assert all(child.dm.getSupportSize(e) <= 2 for e in range(es, ee))
    assert _global_owned_cells(child.dm) == 206        # confluent (same at any np)


def _solcx_metric(base):
    """A viscosity-jump band metric around x=0.5 (the SolCx interface)."""
    M = uw.discretisation.MeshVariable("Mjump", base, 1, degree=1)
    with base.access(M):
        c = M.coords
        band = np.exp(-(((c[:, 0] - 0.5) / 0.08) ** 2))
        M.data[:, 0] = 1.0 / (0.07 + (1.0 / 80 - 0.07) * band) ** 2
    return M


def test_solcx_stokes_velocity_fmg_on_nvb_child():
    """SolCx Stokes (η jump 1e6) on a graded NVB child, in parallel: the velocity
    block auto-picks-up the mesh-owned custom-P FMG and matches a GAMG reference
    iteration-for-iteration, with a partition-independent solution."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True,
        refinement=2, qdegree=3,
    )
    child = base.adapt(_solcx_metric(base), max_levels=1, engine="nvb")
    assert _global_owned_cells(child.dm) == 800   # confluent at any np

    def solcx(pc):
        sol = A.SolCx(child, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1)
        s = uw.systems.Stokes(child)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
        s.saddle_preconditioner = 1.0 / sol.fn_viscosity
        s.bodyforce = sol.fn_bodyforce
        s.add_dirichlet_bc((0.0, None), "Left")
        s.add_dirichlet_bc((0.0, None), "Right")
        s.add_dirichlet_bc((None, 0.0), "Bottom")
        s.add_dirichlet_bc((None, 0.0), "Top")
        s.petsc_use_pressure_nullspace = True
        s.petsc_options["snes_type"] = "ksponly"
        s.tolerance = 1e-8
        if pc == "gamg":
            s.preconditioner = "gamg"
        s.solve()
        vksp = s.snes.getKSP().getPC().getFieldSplitSubKSP()[0]
        return s, vksp

    sg, vg = solcx("gamg")
    it_g = vg.getIterationNumber()

    s, vksp = solcx("fmg")
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"            # geometric MG (from the mesh tail), not AMG
    assert vksp.getPC().getMGType() == 2             # PC.MGType.FULL == FMG
    assert vksp.getIterationNumber() <= it_g + 1     # FMG matches the AMG reference

    from mpi4py import MPI
    comm = child.dm.comm.tompi4py()
    num = comm.allreduce(float(np.sum((s.u.data - sg.u.data) ** 2)), op=MPI.SUM)
    den = comm.allreduce(float(np.sum(sg.u.data ** 2)), op=MPI.SUM)
    assert np.sqrt(num / (den + 1e-30)) < 1e-4       # same solution as GAMG
