"""Consistent Boundary Flux (solver.boundary_flux) — scalar surface heat flux.

The essential-BC reaction of a diffusion solve, de-smeared by the boundary mass, is the
consistent surface flux -k dT/dn (Gresho et al.); its boundary mean is the Nusselt
number. Validated against a harmonic manufactured solution with a known analytic flux.
The nodal reaction is PETSc's (rock-solid) volume FEM residual; its complete value at a
partition-cut boundary node is assembled by summing each rank's partial contribution by
coordinate, so the flux is partition-independent (see the parallel test).
"""
import numpy as np
import sympy
import pytest
import underworld3 as uw
from underworld3 import analytic as A

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _heatflux_diagnostics(res=48):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("Tbf", mesh, 1, degree=2)
    q = uw.discretisation.MeshVariable("qbf", mesh, 1, degree=1, continuous=True)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Left")
    poisson.add_dirichlet_bc(0.0, "Right")
    poisson.add_dirichlet_bc(sympy.Matrix([sympy.sin(sympy.pi * x)]), "Top")
    poisson.tolerance = 1e-11
    poisson.petsc_options["snes_type"] = "ksponly"
    poisson.solve()

    # general API: boundary flux + field hand-off
    xs, flux = poisson.boundary_flux("Bottom")               # lumped, mean kept
    poisson.boundary_flux_field("Bottom", q)
    # field is symbolically usable
    bd_q = float(uw.maths.BdIntegral(mesh=mesh, fn=q.sym[0], boundary="Bottom").evaluate())
    direct_integral = poisson.boundary_flux_integral("Bottom")

    xc = np.asarray(xs)[:, 0] if len(xs) else np.zeros(0)
    q_an = np.pi * np.sin(np.pi * xc) / np.sinh(np.pi)       # analytic outward flux
    return np.asarray(flux), q_an, bd_q, direct_integral


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial diagnostic: rank-local flux norms are 0/0 on non-owning ranks")
def test_boundary_flux_scalar_heatflux_serial():
    """Surface heat flux reproduces the analytic flux to high accuracy, and its mean is
    the (analytic) Nusselt number — NOT removed."""
    flux, q_an, bd_q, direct_integral = _heatflux_diagnostics(res=48)
    corr = np.dot(flux, q_an) / (np.linalg.norm(flux) * np.linalg.norm(q_an))
    fa = flux if corr >= 0 else -flux
    relL2 = np.linalg.norm(fa - q_an) / np.linalg.norm(q_an)
    assert abs(corr) > 0.999, f"heat flux corr {corr:.4f} too low"
    assert relL2 < 0.01, f"heat flux relL2 vs analytic {relL2:.4f} too large"
    # physical MEAN flux preserved (Nusselt): |mean| ≈ 2/sinh(pi)
    assert np.isclose(abs(fa.mean()), 2.0 / np.sinh(np.pi), rtol=0.02), (
        f"mean flux {fa.mean():.4f} != Nusselt {2.0/np.sinh(np.pi):.4f}")
    assert abs(bd_q) > 0.0                                    # field populated + usable
    # The reaction sum is the integral itself, whereas integrating the recovered
    # pointwise field includes its finite-resolution projection error.
    assert np.isclose(
        abs(direct_integral), 2.0 / np.sinh(np.pi), rtol=1.0e-7, atol=0.0
    )


def _uniform_flux_3d(degree, mass):
    """Unit-cube conduction with exact pointwise outward flux -1 on Bottom."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.45,
        regular=True,
        qdegree=3,
    )
    temperature = uw.discretisation.MeshVariable(
        f"Tbf3d_p{degree}", mesh, 1, degree=degree
    )
    poisson = uw.systems.Poisson(mesh, u_Field=temperature)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(1.0, "Top")
    poisson.tolerance = 1.0e-11
    poisson.petsc_options["snes_type"] = "ksponly"
    poisson.solve()
    xs, flux = poisson.boundary_flux("Bottom", mass=mass)
    return poisson, np.asarray(xs), np.asarray(flux)


@pytest.mark.level_2
@pytest.mark.parametrize(("degree", "mass"), ((1, "lumped"), (2, "auto")))
def test_boundary_flux_3d_pointwise_uniform_serial(degree, mass):
    """P1 and P2 recovery reproduce constant flux at every triangular-trace node."""
    _poisson, _xs, flux = _uniform_flux_3d(degree, mass)
    assert np.allclose(flux, -1.0, rtol=0.0, atol=1.0e-8)


@pytest.mark.level_2
def test_boundary_flux_3d_p2_lumped_rejected():
    """P2 triangle vertex row sums are zero, so nodal lumping is not pointwise valid."""
    poisson, _xs, _flux = _uniform_flux_3d(2, "consistent")
    with pytest.raises(ValueError, match="zero row-sum mass"):
        poisson.boundary_flux("Bottom", mass="lumped")


if __name__ == "__main__":
    _f, _a, _b, _i = _heatflux_diagnostics()
    c = np.dot(_f, _a) / (np.linalg.norm(_f) * np.linalg.norm(_a))
    print(f"corr={abs(c):.4f} relL2={np.linalg.norm((_f if c>=0 else -_f)-_a)/np.linalg.norm(_a):.4f}")


def test_boundary_flux_inhomogeneous_dirichlet_wall():
    """#407: flux through a g != 0 Dirichlet wall must be exact.

    The reaction was previously evaluated against a state whose constrained
    DOFs were zero-filled (the global vector carries no constrained DOFs and
    the round-trip stripped them), so the g=1 wall read ~-17.7 for a true
    unit flux while the g=0 wall was accidentally exact.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    T = uw.discretisation.MeshVariable("T407", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    # Side walls are natural (zero flux), so every node — end nodes
    # included — must read the exact unit flux, outward-normal signed.
    for wall, sign in (("Top", -1.0), ("Bottom", +1.0)):
        _xs, flux = poisson.boundary_flux(wall)
        flux = np.asarray(flux)
        assert np.allclose(flux, sign, atol=1e-3), (
            f"{wall}: flux range [{flux.min()}, {flux.max()}], expected {sign}"
        )


def test_boundary_flux_corner_semantics_all_walls_driven():
    """Corner reactions integrate flux through BOTH adjacent driven walls.

    T = x + y (exact in P2) with all four walls Dirichlet: interior wall
    nodes read the exact unit flux; a corner node reads the SUM of the two
    walls' contributions (0 on one diagonal, 2 on the other) divided by the
    queried wall's mass. This pins the method's documented corner
    semantics — consumers reading pointwise flux at a two-Dirichlet-wall
    corner must expect the mixture (see issue #407 discussion)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    T = uw.discretisation.MeshVariable("T407c", mesh, 1, degree=2)
    x, y = mesh.X
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for wall in ("Top", "Bottom", "Left", "Right"):
        poisson.add_dirichlet_bc(sympy.Matrix([x + y]), wall)
    poisson.solve()

    xs, flux = poisson.boundary_flux("Top")
    xs = np.asarray(xs)
    flux = np.asarray(flux)
    interior = (xs[:, 0] > 1e-6) & (xs[:, 0] < 1.0 - 1e-6)
    assert np.allclose(flux[interior], 1.0, atol=1e-3)
    left_corner = np.isclose(xs[:, 0], 0.0)
    right_corner = np.isclose(xs[:, 0], 1.0)
    # The Top corner reaction mixes the adjacent wall's flux: cancellation
    # at Top-Left (opposite outward fluxes), doubling at Top-Right.
    assert np.allclose(flux[left_corner], 0.0, atol=1e-3)
    assert np.allclose(flux[right_corner], 2.0, atol=1e-3)


@pytest.mark.parametrize("mass", ("auto", "consistent"))
def test_boundary_flux_p1_trace_2d(mass):
    """#413: a 2D P1 trace has no edge-midpoint DOFs — recovery previously died
    with a bare KeyError assembling P2 line masses. T = 1 - y is exact in P1, so
    every wall node must read the exact unit flux with the P1 line mass."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    T = uw.discretisation.MeshVariable("T413", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()

    for wall, sign in (("Top", -1.0), ("Bottom", +1.0)):
        _xs, flux = poisson.boundary_flux(wall, mass=mass)
        flux = np.asarray(flux)
        assert np.allclose(flux, sign, atol=1e-3), (
            f"{wall} (mass={mass}): flux range [{flux.min()}, {flux.max()}]"
        )


def _unit_flux_2d(degree, res=8):
    """Unit-box conduction, T = 1 - y: exact at every degree, unit flux on Top/Bottom."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(res, res), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=4)
    T = uw.discretisation.MeshVariable(f"T459_{degree}", mesh, 1, degree=degree)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc(1.0, "Bottom")
    poisson.add_dirichlet_bc(0.0, "Top")
    poisson.solve()
    return poisson


@pytest.mark.parametrize("degree", (1, 2, 3))
def test_boundary_flux_degree_sweep_2d(degree):
    """#459: the trace degree must not change the answer. T = 1 - y is exact at every
    degree, so each wall node must read the exact unit flux. A degree-3 trace carries
    TWO edge-interior reactions per edge; both were keyed by the edge's single
    coordinate, collapsed onto one dictionary key, and the recovery silently returned
    0.57-0.74 of the unit flux from 17 of the 25 trace nodes."""
    poisson = _unit_flux_2d(degree)
    for wall, sign in (("Top", -1.0), ("Bottom", +1.0)):
        xs, flux = poisson.boundary_flux(wall)
        assert np.allclose(np.asarray(flux), sign, atol=1e-3), (
            f"{wall} (degree={degree}): flux range "
            f"[{np.min(flux)}, {np.max(flux)}], expected {sign}")
        if uw.mpi.size == 1:
            # every trace node must report — the collapse also DROPPED nodes
            assert len(np.asarray(xs)) == degree * 8 + 1


def test_boundary_flux_p3_interior_node_placement():
    """#459: each degree-3 edge-interior reaction pairs with its OWN node coordinate.
    T = xy gives dT/dn = x on Top, so the recovered flux is linear in x; swapping the
    two interior nodes of an edge (or mis-placing them at the midpoint) displaces the
    flux by O(edge/3) ~ 2e-2 at this resolution. Corner columns are excluded: corner
    reactions mix both driven walls and the consistent mass spreads that mixture over
    about one element (documented corner semantics)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=4)
    T = uw.discretisation.MeshVariable("T459xy", mesh, 1, degree=3)
    x, y = mesh.X
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for wall in ("Top", "Bottom", "Left", "Right"):
        poisson.add_dirichlet_bc(sympy.Matrix([x * y]), wall)
    poisson.solve()

    xs, flux = poisson.boundary_flux("Top")          # auto -> consistent at degree 3
    xs = np.asarray(xs)
    flux = np.asarray(flux)
    mid = (xs[:, 0] > 0.35) & (xs[:, 0] < 0.65)
    if uw.mpi.size == 1:
        assert np.count_nonzero(mid) > 0
    assert np.allclose(flux[mid], xs[mid, 0], atol=5e-3), (
        f"mid-wall flux error {np.max(np.abs(flux[mid] - xs[mid, 0])):.3e} — "
        "degree-3 interior nodes mis-placed or mis-ordered")


def test_boundary_flux_p3_collapse_guard(monkeypatch):
    """#459 negative control: force the pre-fix behaviour (both edge-interior nodes
    keyed by the edge's one coordinate) and the de-smear must REFUSE — the silent
    overwrite is exactly what returned wrong flux before the fix."""
    from underworld3.utilities import boundary_flux as bf

    poisson = _unit_flux_2d(3, res=4)
    true_coords = bf._trace_interior_coords

    def collapsed(solver, degree):
        full = true_coords(solver, degree)
        return {e: np.repeat(a.mean(axis=0, keepdims=True), len(a), axis=0)
                for e, a in full.items()}

    monkeypatch.setattr(bf, "_trace_interior_coords", collapsed)
    with pytest.raises(RuntimeError, match="collapse"):
        poisson.boundary_flux("Top")


def test_boundary_flux_scalar_reaction_needs_no_field_decomposition():
    """A scalar reaction scatters its sole field without allocating a sub-DM."""
    poisson = _unit_flux_2d(1, res=4)

    poisson.boundary_flux("Top")
    poisson.boundary_flux("Bottom")

    assert not poisson._subdict


def test_volume_residual_fields_insert_essential_values():
    """#411: compute_volume_residual_fields (Stokes-only diagnostic) missed the
    #407 insert — its residual must now match _assemble_volume_reaction (the
    fixed, validated core) exactly, including on g != 0 Dirichlet walls where the
    un-inserted version returned garbage at constrained rows."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("v411", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p411", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    # v = (x, -y): divergence-free, exact in P2, g != 0 on every wall.
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc(sympy.Matrix([x, -y]).T, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()

    reference = np.asarray(stokes._assemble_volume_reaction())
    out = stokes.compute_volume_residual_fields()

    # Extract each field's slice of the reference using the same local-section
    # walk the method uses, and require exact agreement.
    local_section = stokes.dm.getLocalSection()
    pStart, pEnd = local_section.getChart()
    for name, var in stokes.fields.items():
        field_id = getattr(var, "_solver_field_id", None)
        if field_id is None:
            field_id = var.field_id
        indices = []
        for point in range(pStart, pEnd):
            dof = local_section.getFieldDof(point, field_id)
            if dof > 0:
                offset = local_section.getFieldOffset(point, field_id)
                indices.extend(range(offset, offset + dof))
        assert np.allclose(out[name], reference[indices], rtol=1e-12, atol=1e-14), (
            f"{name}: compute_volume_residual_fields diverges from "
            f"_assemble_volume_reaction on g != 0 walls — essential values not inserted"
        )


@pytest.mark.level_2
def test_cbf_refuses_a_multiplier_constrained_boundary():
    """CBF must not silently return a constant on a constrained boundary (#614).

    The back-calculation reads the VELOCITY residual. Where `add_constraint_bc`
    holds the boundary, the traction has been moved into the multiplier block and
    the velocity rows retain only a constant one — the reaction comes back exactly
    proportional to the nodal boundary mass (ratio 4:2:1 at midpoint / interior
    vertex / end vertex, the P2 line lumping), so de-smearing returns that constant.
    Measured std across the boundary was 2e-15 where the equivalent `Stokes` solve
    varied correctly. The danger is that the value LOOKS reasonable.

    The unconstrained boundary of the same solver must keep working, or this is a
    guard on the solver rather than on the boundary.
    """
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(16, 16), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=1.0e3, x_c=0.5, n=1)
    solver = uw.systems.Stokes_Constrained(mesh)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    solver.bodyforce = sol.fn_bodyforce
    solver.tolerance = 1.0e-9
    solver.petsc_use_pressure_nullspace = True
    solver.add_constraint_bc(0.0, "Top")
    solver.add_dirichlet_bc((None, 0.0), "Bottom")
    solver.add_dirichlet_bc((0.0, None), "Left")
    solver.add_dirichlet_bc((0.0, None), "Right")
    solver.solve()

    with pytest.raises(NotImplementedError, match="add_constraint_bc"):
        solver.boundary_flux("Top", normal=[[0.0, 1.0]])

    # negative control: the Dirichlet boundary of the SAME solver still recovers,
    # and recovers something with spatial content rather than a constant.
    xs, flux = solver.boundary_flux("Bottom", normal=[[0.0, -1.0]])
    assert len(xs) > 0
    assert np.asarray(flux).std() > 1.0e-3, (
        "the unconstrained boundary came back constant too — the guard is on the "
        "solver, not on the boundary")
